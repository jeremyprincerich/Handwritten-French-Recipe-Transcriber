#!/usr/bin/env python3
"""
Handwritten Recipe Notebook Transcription Tool

Transcribes handwritten French cooking recipe notebook images into a clean,
beautifully formatted PDF using Gemini + Ollama vision models.

Model strategy:
  1. Gemini 3 Flash (Google API, fast & accurate, rate-limited free tier)
  2. Ollama qwen3.5:397b-cloud (fallback when Gemini is rate-limited)
  3. Gemini 2.0 Flash (final fallback, highly capable free tier)

Pipeline:
  1. Image preprocessing (CLAHE contrast, sharpen, upscale)
  2. Transcription with model fallback chain
  3. Validation guardrails (detects hallucination, saves progress after each page)
  4. Pages joined seamlessly (recipes spanning pages merge naturally)
  5. Renders to a professional cookbook-style PDF via WeasyPrint

Background mode:
  Use --background to run in the background, surviving terminal closure
  and preventing macOS sleep. Logs go to transcription.log.
"""

import sys
import os
import io
import re
import json
import base64
import time
import subprocess
import logging
import signal
import platform
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import numpy as np
import ollama
from google import genai
from google.genai import types
from PIL import Image, ImageEnhance, ImageFilter
from natsort import natsorted
import markdown
from weasyprint import HTML, CSS

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model chain: Gemini first (prefixed with "gemini:"), then Ollama models.
# Gemini models are called via Google GenAI SDK.
# Ollama models are called via Ollama SDK.
DEFAULT_MODEL_CHAIN = [
    "gemini:gemini-3-flash-preview",
    "ollama:qwen3.5:397b-cloud",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

TRANSCRIPTION_PROMPT = """\
You are an expert French culinary archivist specializing in reading handwritten recipe notebooks.
The page you are given is from a **French cooking recipe notebook** (cahier de recettes).
All text is in French. Your job is to transcribe it faithfully.

Domain context to help you decipher difficult words:
- This notebook contains **recettes de cuisine** (cooking recipes).
- Expect to see: recipe titles, ingredient lists with quantities, cooking instructions/steps.
- Common French culinary measurements: cuillère à soupe (c. à s.), cuillère à café (c. à c.),
  grammes (g), kilogrammes (kg), litres (L), millilitres (mL), centilitres (cL), pincée, verre, tasse.
- Common French cooking verbs: mélanger, fouetter, incorporer, pétrir, laisser reposer,
  préchauffer, enfourner, faire revenir, émincer, hacher, blanchir, mijoter, dorer, etc.
- Common French ingredients: farine, sucre, beurre, œufs, lait, crème fraîche, levure,
  sel, poivre, oignon, ail, persil, thym, laurier, huile d'olive, vinaigre, bouillon, etc.

Transcription rules:
1. Output clean Markdown.
2. Use **### Recipe Title** (heading level 3) for each recipe name you find on the page.
3. Format ingredient lists as Markdown bullet lists (- item).
4. Format cooking steps as numbered lists (1. Step one, 2. Step two, etc.).
5. Do NOT translate, summarize, autocorrect, or paraphrase. Keep every word as-is in French.
6. If a word is difficult to read, use culinary context to make your best guess.
   Only mark a word as [illisible] if absolutely impossible to decipher.
7. Preserve French accented characters exactly: é, è, ê, ë, à, â, ù, û, ô, î, ï, ç, etc.
8. Output ONLY the transcribed Markdown text. No preamble, no commentary, no code fences.
9. Do NOT use any thinking/reasoning tags. Output the recipe text directly.
10. If the page starts mid-sentence or mid-recipe (a continuation from a previous page),
    just transcribe what you see. Do NOT add a title or heading if one isn't written on the page."""

PDF_CSS = '''
@page { 
    size: A4; 
    margin: 2.5cm 2cm; 
    @bottom-right {
        content: counter(page);
        font-family: Georgia, serif;
        font-size: 10pt;
        color: #666;
    }
}
body { font-family: Georgia, "Times New Roman", Times, serif; font-size: 11pt; line-height: 1.65; color: #2c2c2c; }
h1 { font-family: Georgia, serif; font-size: 28pt; text-align: center; color: #8B0000; margin-bottom: 0.3em; letter-spacing: 0.05em; }
h1 + p { text-align: center; font-style: italic; color: #666; margin-bottom: 2em; }
h2 { color: #555; font-size: 11pt; font-weight: normal; font-style: italic; border-bottom: none; margin-top: 1em; margin-bottom: 0.5em; }
h3 { font-family: Georgia, serif; font-size: 16pt; color: #8B0000; border-bottom: 2px solid #D4A574; padding-bottom: 0.2em; margin-top: 1.5em; margin-bottom: 0.8em; }
strong { color: #5C3317; }
ul { margin: 0.8em 0; padding-left: 1.5em; }
ul li { margin-bottom: 0.3em; line-height: 1.5; }
ol { margin: 0.8em 0; padding-left: 1.5em; }
ol li { margin-bottom: 0.5em; line-height: 1.5; }
p { margin-bottom: 0.8em; }
hr { border: none; border-top: 1px dashed #D4A574; margin: 1.5em 0; }
'''

# =============================================================================
# LOGGING
# =============================================================================

log = logging.getLogger("transcribe")


def setup_logging(log_file: str | None = None):
    """Configure logging to console and optionally to a file."""
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    log.setLevel(logging.INFO)

    # Always log to console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    log.addHandler(ch)

    # Optionally log to file
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)


# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================

def preprocess_image(image_path: Path, target_long_edge: int = 2048) -> bytes:
    """Preprocess a handwritten notebook image. Returns JPEG bytes."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            w, h = img.size
            long_edge = max(w, h)
            if long_edge < target_long_edge:
                scale = target_long_edge / long_edge
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

            gray = img.convert("L")
            gray = _apply_clahe(gray, clip_limit=2.0, tile_size=32)
            gray = gray.filter(ImageFilter.SHARPEN)
            gray = gray.filter(ImageFilter.SHARPEN)
            gray = gray.filter(ImageFilter.MedianFilter(size=3))
            enhancer = ImageEnhance.Contrast(gray)
            gray = enhancer.enhance(1.4)
            final = gray.convert("RGB")

            buffer = io.BytesIO()
            final.save(buffer, format="JPEG", quality=90)
            return buffer.getvalue()
    except Exception as e:
        raise RuntimeError(f"Failed to preprocess {image_path.name}: {e}")


def _apply_clahe(pil_gray: Image.Image, clip_limit: float = 2.0, tile_size: int = 32) -> Image.Image:
    """Contrast-Limited Adaptive Histogram Equalization (numpy-only)."""
    arr = np.array(pil_gray, dtype=np.float64)
    h, w = arr.shape
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size
    padded = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="edge")
    ph, pw = padded.shape
    tiles_y, tiles_x = ph // tile_size, pw // tile_size
    result = np.zeros_like(padded)

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            y0, y1 = ty * tile_size, (ty + 1) * tile_size
            x0, x1 = tx * tile_size, (tx + 1) * tile_size
            tile = padded[y0:y1, x0:x1]
            hist, _ = np.histogram(tile.flatten(), bins=256, range=(0, 256))
            limit = int(clip_limit * tile.size / 256)
            excess = 0
            for i in range(256):
                if hist[i] > limit:
                    excess += hist[i] - limit
                    hist[i] = limit
            hist += excess // 256
            cdf = hist.cumsum()
            cdf_min = cdf[cdf > 0].min() if cdf[cdf > 0].size > 0 else 0
            denom = tile.size - cdf_min
            if denom == 0:
                result[y0:y1, x0:x1] = tile
            else:
                lut = ((cdf - cdf_min) / denom * 255).clip(0, 255).astype(np.uint8)
                result[y0:y1, x0:x1] = lut[tile.astype(np.uint8)]

    return Image.fromarray(result[:h, :w].astype(np.uint8))


# =============================================================================
# VALIDATION
# =============================================================================

def validate_transcription(text: str, image_name: str) -> tuple[bool, str]:
    """Validate that a transcription looks legitimate and not hallucinated."""
    if len(text.strip()) < 30:
        return False, "too short (< 30 chars)"

    plain = re.sub(r'[#*\-\d\.\[\]]', '', text).strip()
    words = plain.lower().split()
    if len(words) > 20:
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        from collections import Counter
        bigram_counts = Counter(bigrams)
        if bigram_counts:
            most_common_count = bigram_counts.most_common(1)[0][1]
            if most_common_count > max(8, len(bigrams) * 0.15):
                return False, f"excessive repetition"

    english_markers = ["the ", " is ", " are ", " was ", " were ", " have ", " has ",
                       " this ", " that ", " with ", " from ", " would ", " could "]
    french_markers = ["le ", "la ", "les ", " de ", " des ", " du ", " un ", " une ",
                      " et ", " ou ", " dans ", " avec ", " pour ", " sur ", " au "]
    lower = text.lower()
    en_count = sum(1 for m in english_markers if m in lower)
    fr_count = sum(1 for m in french_markers if m in lower)
    if en_count > 5 and fr_count < 2:
        return False, f"appears English, not French"

    hallucination_patterns = [
        r"(?i)as an ai", r"(?i)i('m| am) (sorry|unable|not able)",
        r"(?i)here('s| is) (a|the|my) (transcription|translation)",
        r"(?i)unfortunately", r"(?i)i can('t|not) (read|see|make out)",
    ]
    for pattern in hallucination_patterns:
        if re.search(pattern, text) and len(text) < 500:
            return False, f"AI commentary detected"

    return True, "OK"


# =============================================================================
# MODEL CALLERS
# =============================================================================

_gemini_client = None
_last_call_time = time.time()  # Track time between calls for sleep detection
SLEEP_GAP_THRESHOLD = 30      # If >30s gap between calls, assume sleep/wake occurred
_gemini_api_keys = []
_current_key_idx = 0

def init_gemini_keys(api_key_str: str):
    global _gemini_api_keys, _current_key_idx
    if api_key_str:
        _gemini_api_keys = [k.strip() for k in api_key_str.split(",") if k.strip()]
    _current_key_idx = 0

def rotate_gemini_key() -> bool:
    global _current_key_idx, _gemini_client
    if len(_gemini_api_keys) > 1:
        _current_key_idx = (_current_key_idx + 1) % len(_gemini_api_keys)
        _gemini_client = None  # Force fresh client
        masked = _gemini_api_keys[_current_key_idx][:6] + "..."
        log.info(f"      🔄 Rotated to next Gemini API Key (starts with {masked})")
        return True
    return False


def _detect_sleep_wake() -> bool:
    """Detect if the system likely slept and woke up since the last API call.
    Returns True if a sleep/wake cycle was detected."""
    global _last_call_time
    now = time.time()
    gap = now - _last_call_time
    _last_call_time = now
    if gap > SLEEP_GAP_THRESHOLD:
        log.info(f"      💤 Detected possible sleep/wake (gap: {gap:.0f}s) — resetting connections")
        return True
    return False


def _reset_connections():
    """Reset all cached clients/connections after a sleep/wake cycle."""
    global _gemini_client
    _gemini_client = None
    log.info(f"      🔄 Connections reset (Gemini client + Ollama will reconnect)")


def _get_gemini_client(api_key: str) -> genai.Client:
    """Lazily create and cache the Gemini client."""
    global _gemini_client
    if _gemini_client is None:
        key = _gemini_api_keys[_current_key_idx] if _gemini_api_keys else api_key
        _gemini_client = genai.Client(api_key=key)
    return _gemini_client


def _check_ollama_health() -> bool:
    """Quick health check to see if Ollama is reachable."""
    try:
        ollama.list()  # Lightweight call to check connectivity
        return True
    except Exception as e:
        log.info(f"      ⚠️  Ollama health check failed: {e}")
        return False


# Per-call timeout in seconds. If Gemini or Ollama hangs (e.g. after sleep),
# the call is killed after this many seconds and we fall back.
API_CALL_TIMEOUT = 180  # 180 seconds per call (397b-cloud often takes 60-90s)
PAGE_TIMEOUT = 600      # 10 minutes total per page (generous for sleep recovery)


def call_gemini(model: str, image_bytes: bytes, prompt: str, api_key: str) -> str:
    """Call Google Gemini API with an image. Returns text or raises."""
    # Detect sleep/wake and reset client if needed
    if _detect_sleep_wake():
        _reset_connections()

    client = _get_gemini_client(api_key)

    def _call():
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                prompt,
            ],
        )
        return response.text.strip() if response.text else ""

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_call)
        try:
            text = future.result(timeout=API_CALL_TIMEOUT)
        except FuturesTimeoutError:
            future.cancel()
            raise TimeoutError(f"Gemini call timed out after {API_CALL_TIMEOUT}s")

    return _strip_thinking_tags(text)


def call_ollama_vision(model: str, image_bytes: bytes, prompt: str) -> str:
    """Call Ollama with an image. Returns text or raises."""
    # Detect sleep/wake and reset connections if needed
    if _detect_sleep_wake():
        _reset_connections()

    # Health check before the heavy call
    if not _check_ollama_health():
        log.info(f"      🔄 Ollama unreachable, waiting 10s for it to recover...")
        time.sleep(10)
        if not _check_ollama_health():
            raise ConnectionError("Ollama is not reachable after sleep/wake recovery")

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    def _call():
        response = ollama.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            }],
        )
        return response.message.content.strip()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_call)
        try:
            text = future.result(timeout=API_CALL_TIMEOUT)
        except FuturesTimeoutError:
            future.cancel()
            raise TimeoutError(f"Ollama call timed out after {API_CALL_TIMEOUT}s")

    return _strip_thinking_tags(text)


def _strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> tags that some models emit."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def get_image_files(directory: str) -> list:
    """Get naturally sorted image files from a directory."""
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    image_files = [
        f for f in dir_path.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return natsorted(image_files, key=lambda p: p.name)


def transcribe_image(
    image_path: Path,
    model_chain: list[str],
    api_key: str = "",
    max_attempts: int = 3,
    max_full_retries: int = 2,
) -> str:
    """
    Transcribe a single image with model fallback, validation, and
    automatic sleep/wake recovery.

    If all models fail (possibly due to sleep), the entire chain is
    retried up to max_full_retries times with a pause in between.
    """
    if not _gemini_api_keys and api_key:
        init_gemini_keys(api_key)

    for full_retry in range(max_full_retries + 1):
        page_start = time.time()
        image_bytes = preprocess_image(image_path)
        best_text = None
        best_len = 0
        all_connection_errors = True  # Track if ALL failures were connection-related

        for model_spec in model_chain:
            # Check overall page timeout
            if time.time() - page_start > PAGE_TIMEOUT:
                log.info(f"      ⏰ Page timeout ({PAGE_TIMEOUT}s) reached, stopping retries")
                break

            provider, model_name = _parse_model_spec(model_spec)
            hit_rate_limit = False

            attempt = 1
            while attempt <= max_attempts * 10:
                # Check overall page timeout
                if time.time() - page_start > PAGE_TIMEOUT:
                    log.info(f"      ⏰ Page timeout ({PAGE_TIMEOUT}s) reached")
                    break

                try:
                    start_time = time.time()

                    if provider == "gemini":
                        if not api_key:
                            log.info(f"      ⚠️  No Gemini API key, skipping {model_name}")
                            break
                        text = call_gemini(model_name, image_bytes, TRANSCRIPTION_PROMPT, api_key)
                    else:
                        text = call_ollama_vision(model_name, image_bytes, TRANSCRIPTION_PROMPT)

                    elapsed = time.time() - start_time
                    log.info(f"      ({provider}:{model_name}, {elapsed:.1f}s, {len(text)} chars)")
                    all_connection_errors = False  # Got a response (even if invalid)

                    # Validate
                    is_valid, reason = validate_transcription(text, image_path.name)
                    if is_valid:
                        return text

                    log.info(f"      ⚠️  Validation failed: {reason}")
                    if len(text) > best_len:
                        best_text = text
                        best_len = len(text)

                    # Retry with heavier preprocessing
                    if attempt == 1:
                        log.info(f"      ↻ Retrying with heavier preprocessing...")
                        image_bytes = preprocess_image(image_path, target_long_edge=3000)

                except TimeoutError as e:
                    log.info(f"      ⏰ {provider}:{model_name} TIMED OUT: {e}")
                    log.info(f"      → Skipping to next model (hung connection)")
                    break  # Move to next model immediately on timeout

                except (ConnectionError, OSError) as e:
                    log.info(f"      🔌 {provider}:{model_name} connection error: {e}")
                    log.info(f"      → Connection lost (likely sleep/wake), will retry chain")
                    _reset_connections()
                    break

                except Exception as e:
                    error_str = str(e).lower()
                    is_rate_limit = any(kw in error_str for kw in [
                        "429", "rate", "quota", "resource_exhausted", "too many"
                    ])
                    is_transient = any(kw in error_str for kw in [
                        "500", "503", "unavailable", "timeout", "connection",
                        "disconnected", "eof", "reset", "broken pipe"
                    ])

                    if is_rate_limit:
                        hit_rate_limit = True
                        rotated = False
                        if provider == "gemini":
                            rotated = rotate_gemini_key()
                        
                        if rotated:
                            log.info(f"      ⚡ {provider}:{model_name} rate limited. Switched API key, retrying...")
                        else:
                            wait_time = 60
                            log.info(f"      ⚡ {provider}:{model_name} rate limited. Waiting {wait_time}s to retry free tier...")
                            time.sleep(wait_time)
                        all_connection_errors = False
                        attempt += 1
                        continue
                    elif is_transient:
                        wait_time = 10 * attempt
                        log.info(f"      ⏳ {provider}:{model_name} transient error ({e}), waiting {wait_time}s...")
                        time.sleep(wait_time)
                        attempt += 1
                        continue
                    else:
                        log.info(f"      ⚠️  {provider}:{model_name} error: {e}")
                        all_connection_errors = False
                        break

                attempt += 1

            if hit_rate_limit:
                log.info(f"    → {provider}:{model_name} rate limited, falling back...")
            else:
                log.info(f"    → Moving from {provider}:{model_name}...")

            # Reset preprocessing for next model
            image_bytes = preprocess_image(image_path)

        if best_text:
            log.info(f"      ⚠️  Using best-effort result ({best_len} chars)")
            return best_text

        # If all failures were connection-related and we have retries left,
        # wait and retry the entire chain (likely a sleep/wake recovery)
        if full_retry < max_full_retries and all_connection_errors:
            wait = 30 * (full_retry + 1)
            log.info(f"      🔄 All models had connection errors — waiting {wait}s then retrying full chain ({full_retry + 1}/{max_full_retries})")
            _reset_connections()
            time.sleep(wait)
            continue
        else:
            break

    raise RuntimeError(f"All models failed for {image_path.name}")


def _parse_model_spec(spec: str) -> tuple[str, str]:
    """Parse 'provider:model' into (provider, model_name)."""
    if spec.startswith("gemini:"):
        return "gemini", spec[len("gemini:"):]
    elif spec.startswith("ollama:"):
        return "ollama", spec[len("ollama:"):]
    else:
        # Default to ollama for backward compatibility
        return "ollama", spec


def save_progress(progress_file: Path, transcriptions: dict):
    """Save transcription progress to a JSON file."""
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(transcriptions, f, ensure_ascii=False, indent=2)


def load_progress(progress_file: Path) -> dict:
    """Load previous transcription progress if it exists."""
    if progress_file.exists():
        with open(progress_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def generate_pdf_document(content: str, output_path: str) -> None:
    """Generate a cookbook-style PDF from Markdown content using WeasyPrint."""
    html_body = markdown.markdown(content, extensions=["extra"])
    full_html = f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="utf-8">
        <title>Cahier de Recettes</title>
    </head>
    <body>
        <h1>Cahier de Recettes</h1>
        {html_body}
    </body>
    </html>
    """
    HTML(string=full_html).write_pdf(output_path, stylesheets=[CSS(string=PDF_CSS)])


def transcribe_directory(
    input_directory: str,
    output_pdf: str,
    model_chain: list[str],
    api_key: str = "",
    delay_between_pages: float = 2.0,
    resume: bool = True,
) -> str:
    """
    Transcribe all images and generate a PDF.

    Features:
      - Saves progress after each page (resume on crash/interrupt/reboot)
      - Validates each transcription
      - Adaptive model fallback
    """
    input_path = Path(input_directory)
    progress_file = input_path / ".transcription_progress.json"

    log.info(f"")
    log.info(f"{'='*60}")
    log.info(f"Cahier de Recettes — Transcription Tool")
    log.info(f"{'='*60}")
    log.info(f"Input directory: {input_directory}")
    log.info(f"Models: {' → '.join(model_chain)}")
    log.info(f"Output PDF: {output_pdf}")
    log.info(f"Gemini API key: {'set' if api_key else 'NOT SET (Gemini models will be skipped)'}")
    log.info(f"Sleep recovery: enabled (gap threshold: {SLEEP_GAP_THRESHOLD}s)")
    log.info(f"Timeouts: per-call={API_CALL_TIMEOUT}s, per-page={PAGE_TIMEOUT}s")
    log.info(f"{'='*60}")
    if not api_key:
        log.info(f"")
        log.info(f"⚠️  WARNING: No Gemini API key set!")
        log.info(f"   Gemini is your fastest/most reliable model.")
        log.info(f"   Set it with: export GEMINI_API_KEY='your-key'")
        log.info(f"   Then re-run with --background")
    log.info(f"")

    try:
        image_files = get_image_files(input_directory)
    except (FileNotFoundError, NotADirectoryError) as e:
        log.info(f"Error: {e}")
        sys.exit(1)

    if not image_files:
        log.info(f"No image files found in {input_directory}")
        sys.exit(1)

    log.info(f"Found {len(image_files)} image(s) to process.")
    log.info(f"")

    # Load previous progress
    done_pages = {}
    if resume:
        done_pages = load_progress(progress_file)
        if done_pages:
            log.info(f"  📂 Resuming: {len(done_pages)} page(s) already done.")
            log.info(f"")

    failed_files = []
    total = len(image_files)

    for i, image_path in enumerate(image_files):
        page_num = i + 1
        page_key = image_path.name

        if page_key in done_pages:
            log.info(f"  [{page_num}/{total}] ⏭️  Skipping {page_key} (already done)")
            continue

        log.info(f"  [{page_num}/{total}] ⏳ Transcribing {page_key}...")

        try:
            text = transcribe_image(image_path, model_chain, api_key=api_key)
            done_pages[page_key] = text
            save_progress(progress_file, done_pages)
            log.info(f"  [{page_num}/{total}] ✅ Done: {page_key} ({len(text)} chars) [saved]")
        except Exception as e:
            failed_files.append((page_key, str(e)))
            log.info(f"  [{page_num}/{total}] ❌ Failed: {page_key}: {e}")

        # Delay between pages
        remaining = total - page_num
        if remaining > 0:
            log.info(f"      💤 {delay_between_pages}s pause ({remaining} left)...")
            time.sleep(delay_between_pages)

    if failed_files:
        log.info(f"")
        log.info(f"⚠️  {len(failed_files)} page(s) failed:")
        for name, err in failed_files:
            log.info(f"  - {name}: {err}")

    # Assemble in filename order
    transcriptions = []
    for image_path in image_files:
        if image_path.name in done_pages:
            transcriptions.append(done_pages[image_path.name])

    full_content = "\n\n".join(transcriptions)

    # PDF generation
    log.info(f"")
    log.info(f"Generating PDF ({len(transcriptions)} pages)...")
    try:
        generate_pdf_document(full_content, output_pdf)
        log.info(f"  ✅ PDF: {output_pdf}")
    except Exception as e:
        log.info(f"  ❌ PDF generation failed: {e}")

    # Cleanup progress file on full success
    if len(transcriptions) == total and not failed_files:
        try:
            progress_file.unlink()
            log.info(f"  🧹 Progress file cleaned up.")
        except OSError:
            pass

    log.info(f"")
    log.info(f"{'='*60}")
    log.info(f"DONE — Pages OK: {len(transcriptions)}/{total}")
    if failed_files:
        log.info(f"         Failed: {len(failed_files)}")
    log.info(f"{'='*60}")
    log.info(f"")

    return full_content


def launch_background(args_list: list[str]):
    """
    Re-launch this script in the background using nohup + caffeinate.
    The process will survive terminal closure and prevent macOS sleep.

    Uses caffeinate -s to assert system-level sleep prevention (more
    aggressive than -i which only prevents idle sleep).
    """
    script_path = os.path.abspath(__file__)
    python_path = sys.executable
    log_file = "transcription.log"

    # Build the command without --background to avoid infinite loop
    cmd_args = [a for a in args_list if a != "--background"]
    cmd = f'caffeinate -s nohup {python_path} {script_path} {" ".join(cmd_args)} --log-file {log_file}'

    log.info(f"")
    log.info(f"🚀 Launching in background mode...")
    log.info(f"   The process will survive terminal closure and prevent macOS sleep.")
    log.info(f"   Using caffeinate -s (system sleep prevention).")
    log.info(f"   If you close the lid, the script will auto-recover on wake.")
    log.info(f"   Logs are written to: {log_file}")
    log.info(f"")
    log.info(f"   To monitor progress:")
    log.info(f"     tail -f {log_file}")
    log.info(f"")
    log.info(f"   To stop:")
    log.info(f"     kill $(pgrep -f 'transcribe.py')")
    log.info(f"")

    # Launch detached — stdout/stderr go to log file only
    with open(log_file, "a") as lf:
        subprocess.Popen(
            cmd,
            shell=True,
            stdout=lf,
            stderr=lf,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    log.info(f"✅ Background process started. You can close this terminal.")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Transcribe a French handwritten recipe notebook to PDF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Basic usage (Gemini first, Ollama fallback)
  export GEMINI_API_KEY="your-key"
  python transcribe.py ./notebook_photos

  # Run in background (survives terminal close + prevents Mac sleep)
  python transcribe.py ./notebook_photos --background

  # Monitor background progress
  tail -f transcription.log

  # Use only Ollama (no Gemini)
  python transcribe.py ./notebook_photos --models ollama:qwen3.5:397b-cloud ollama:llama3.2-vision

  # Force fresh start (ignore saved progress)
  python transcribe.py ./notebook_photos --no-resume
"""
    )
    parser.add_argument("input_directory", help="Directory containing notebook images")
    parser.add_argument("--output-pdf", default="transcription.pdf",
                        help="Output PDF path (default: transcription.pdf)")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODEL_CHAIN,
                        help="Model chain: 'gemini:name' and/or 'ollama:name'")
    parser.add_argument("--api-key", default=os.getenv("GEMINI_API_KEY", ""),
                        help="Gemini API key (default: env GEMINI_API_KEY)")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Seconds between pages (default: 2)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore saved progress, re-transcribe everything")
    parser.add_argument("--background", action="store_true",
                        help="Run in background (survives terminal close, prevents Mac sleep)")
    parser.add_argument("--log-file", default=None,
                        help="Log file path (used internally for background mode)")

    args = parser.parse_args()

    # Background mode: re-launch self and exit
    if args.background:
        setup_logging()  # Log to console only for the launch message
        raw_args = sys.argv[1:]
        launch_background(raw_args)
        return

    # Normal mode (or background child): log to file only if --log-file is set
    setup_logging(args.log_file)

    transcribe_directory(
        input_directory=args.input_directory,
        output_pdf=args.output_pdf,
        model_chain=args.models,
        api_key=args.api_key,
        delay_between_pages=args.delay,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
