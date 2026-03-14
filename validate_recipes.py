#!/usr/bin/env python3
"""
Recipe JSON Validator & Corrector

Reads the parsed recipes.json, sends each recipe to an LLM for review,
and outputs a corrected version with errors fixed.

Things the LLM checks for:
  - Garbled / nonsensical text in titles, ingredients, or steps
  - OCR / transcription artefacts (random characters, broken words)
  - Ingredients placed in the preparation list and vice-versa
  - Duplicate items within a single recipe
  - Missing accents or broken French characters
  - Unreasonable quantities (e.g. "3000 kg de beurre")
  - Non-recipe entries that slipped through

Model fallback chain:
  1. OpenRouter: StepFun Step 3.5 Flash (free)
  2. Ollama: qwen3.5:397b-cloud (strong cloud fallback)

Usage:
  python validate_recipes.py recipes.json
  python validate_recipes.py recipes.json --output recipes_corrected.json
  python validate_recipes.py recipes.json --resume
"""

import sys
import os
import re
import json
import time
import argparse
import urllib.request
import urllib.error
from pathlib import Path

try:
    import ollama
except ImportError:
    ollama = None


# =============================================================================
# CONFIGURATION
# =============================================================================

BATCH_SIZE = 5  # Recipes per LLM call

VALIDATION_PROMPT = """\
Tu es un correcteur expert de données culinaires françaises.
Tu reçois un lot de recettes de cuisine en JSON. Chaque recette a: "titre", "ingredients", "preparation", "notes", "numero".

Ton travail est de CORRIGER les erreurs suivantes si tu en trouves :
1. Texte incohérent ou sans rapport avec la cuisine (charabia, artefacts OCR, caractères aléatoires).
2. Ingrédients placés dans la liste "preparation" et vice-versa — remets-les au bon endroit.
3. Doublons exacts dans les listes d'ingrédients ou d'étapes — supprime les doublons.
4. Accents français cassés ou manquants (ex: "creme" → "crème", "oeufs" → "œufs").
5. Quantités manifestement absurdes (ex: "3000 kg de beurre").
6. Entrées qui ne sont clairement PAS des recettes (listes de courses, titres de sections, etc.) — marque-les avec "supprimer": true.
7. Titres vides ou génériques comme "Sans titre" — essaie de deviner un titre à partir du contenu.

RÈGLES :
- Retourne UNIQUEMENT un tableau JSON valide des recettes corrigées (même format d'entrée).
- Si une recette est correcte, retourne-la telle quelle SANS modification.
- NE traduis RIEN en anglais. Tout doit rester en français.
- NE supprime PAS de recettes valides. Ajoute seulement "supprimer": true aux fausses entrées.
- NE fusionne PAS de recettes entre elles.
- Aucun texte, aucune explication. UNIQUEMENT le tableau JSON.
"""


# =============================================================================
# MODEL CALLERS
# =============================================================================

def call_openrouter(model_name: str, prompt: str, data_text: str, api_key: str) -> str:
    """Call OpenRouter API. Returns raw text response."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = json.dumps({
        "model": model_name,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": data_text}
        ],
        "temperature": 0.1,
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    return result["choices"][0]["message"]["content"].strip()


def call_ollama(model_name: str, prompt: str, data_text: str) -> str:
    """Call Ollama with text. Returns raw text response."""
    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": data_text}
        ],
        options={"temperature": 0.1}
    )
    return response.message.content.strip()


def _parse_response(raw: str) -> list[dict] | None:
    """Parse the LLM response into a list of recipe dicts."""
    raw = raw.strip()

    # Strip <think>...</think> tags
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

    # Strip markdown code fences
    raw = re.sub(r'^```(json)?\s*\n?', '', raw)
    raw = re.sub(r'\n?```\s*$', '', raw)
    raw = raw.strip()

    data = json.loads(raw)

    # Handle wrapped format {"recettes": [...]}
    if isinstance(data, dict) and "recettes" in data:
        data = data["recettes"]

    if not isinstance(data, list):
        return None

    return data


def validate_batch(
    recipes_batch: list[dict],
    openrouter_key: str = "",
    models: list[tuple[str, str]] | None = None,
) -> list[dict]:
    """
    Send a batch of recipes to the LLM for validation/correction.
    Returns the corrected batch.
    """
    if models is None:
        models = [
            ("openrouter", "stepfun/step-3.5-flash:free"),
            ("ollama", "qwen3.5:397b-cloud"),
        ]

    batch_json = json.dumps(recipes_batch, ensure_ascii=False, indent=2)

    for provider, model_name in models:
        try:
            start_time = time.time()

            if provider == "openrouter":
                if not openrouter_key:
                    continue
                raw = call_openrouter(model_name, VALIDATION_PROMPT, batch_json, openrouter_key)
            elif provider == "ollama":
                if ollama is None:
                    continue
                raw = call_ollama(model_name, VALIDATION_PROMPT, batch_json)
            else:
                continue

            elapsed = time.time() - start_time

            corrected = _parse_response(raw)
            if corrected is None:
                print(f"        [{provider}:{model_name}] ({elapsed:.1f}s) ✗ invalid response")
                continue

            print(f"        [{provider}:{model_name}] ({elapsed:.1f}s) ✓ ({len(corrected)} recipes)")
            return corrected

        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = any(kw in error_str for kw in [
                "429", "rate", "quota", "resource_exhausted", "too many"
            ])
            if is_rate_limit:
                print(f"        [{provider}:{model_name}] ⚡ rate limited, trying next...")
            else:
                print(f"        [{provider}:{model_name}] ✗ error: {e}")
            continue

    # If all models fail, return the original batch unchanged
    print(f"        ⚠️  All models failed — keeping original batch")
    return recipes_batch


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate and correct a recipe JSON database using an LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python validate_recipes.py recipes.json
  python validate_recipes.py recipes.json --output recipes_final.json
  python validate_recipes.py recipes.json --resume
"""
    )
    parser.add_argument("input_file", help="Path to the recipes JSON file")
    parser.add_argument("--output", "-o", default=None,
                        help="Output JSON file path (default: recipes_corrected.json)")
    parser.add_argument("--openrouter-key", default=os.getenv("OPENROUTER_API_KEY", ""),
                        help="OpenRouter API key (default: env OPENROUTER_API_KEY)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Recipes per LLM call (default: {BATCH_SIZE})")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from saved progress")

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: File not found: {args.input_file}")
        sys.exit(1)

    # Load input
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    recipes = data.get("recettes", [])
    total = len(recipes)

    print(f"\n{'='*60}")
    print(f"Recipe JSON Validator & Corrector")
    print(f"{'='*60}")
    print(f"Input: {args.input_file}")
    print(f"Recipes: {total}")
    print(f"Batch size: {args.batch_size}")
    print(f"OpenRouter key: {'set' if args.openrouter_key else 'NOT SET'}")
    print(f"{'='*60}\n")

    # Progress saving
    progress_file = Path(".validation_progress.json")
    corrected_recipes = []
    start_batch = 0

    if args.resume and progress_file.exists():
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
                corrected_recipes = saved.get("corrected", [])
                start_batch = saved.get("next_batch_start", 0)
                print(f"📂 Resuming: {len(corrected_recipes)} recipes already validated, "
                      f"starting from index {start_batch}\n")
        except Exception as e:
            print(f"Warning: Could not resume: {e}")

    # Process in batches
    print("Starting LLM Validation...\n")

    total_removed = 0
    total_modified = 0

    for i in range(start_batch, total, args.batch_size):
        batch = recipes[i:i + args.batch_size]
        batch_end = min(i + args.batch_size, total)
        batch_num = (i // args.batch_size) + 1
        total_batches = (total + args.batch_size - 1) // args.batch_size

        titles = [r.get("titre", "?")[:40] for r in batch]
        print(f"  [{batch_num}/{total_batches}] Recipes {i+1}–{batch_end}: {', '.join(titles)}")

        corrected_batch = validate_batch(batch, openrouter_key=args.openrouter_key)

        # Track changes
        for orig, corr in zip(batch, corrected_batch):
            if corr.get("supprimer"):
                total_removed += 1
                print(f"      🗑️  Marked for removal: {corr.get('titre', '?')}")
            elif json.dumps(orig, ensure_ascii=False) != json.dumps(corr, ensure_ascii=False):
                total_modified += 1
                print(f"      ✏️  Corrected: {corr.get('titre', '?')}")

        corrected_recipes.extend(corrected_batch)

        # Save progress
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump({
                "next_batch_start": batch_end,
                "corrected": corrected_recipes,
            }, f, ensure_ascii=False)

    # Filter out removed entries and renumber
    print(f"\n--- Post-processing ---")
    final_recipes = []
    for r in corrected_recipes:
        if r.get("supprimer"):
            continue
        r.pop("supprimer", None)
        r["numero"] = len(final_recipes) + 1
        final_recipes.append(r)

    removed_count = len(corrected_recipes) - len(final_recipes)
    print(f"  Recipes validated: {len(corrected_recipes)}")
    print(f"  Removed (not real recipes): {removed_count}")
    print(f"  Corrected: {total_modified}")
    print(f"  Final count: {len(final_recipes)}")

    # Save output
    output_path = args.output or "recipes_corrected.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_recettes": len(final_recipes),
            "recettes": final_recipes,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  ✅ Saved: {output_path} ({len(final_recipes)} recettes)")

    # Clean up progress
    try:
        if progress_file.exists():
            progress_file.unlink()
    except OSError:
        pass

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
