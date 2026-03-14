#!/usr/bin/env python3
"""
Cookbook Generator

Converts a recipes JSON database into a beautifully formatted HTML cookbook
that can be opened, edited, and printed from any browser or word processor.

No AI is used — this is purely deterministic formatting.

Output:
  - cookbook.html: A richly styled, print-ready HTML cookbook
    (open in browser → File → Print → Save as PDF, or edit directly)

Usage:
  python generate_cookbook.py recipes_final.json
  python generate_cookbook.py recipes_final.json --output mon_cahier.html
  python generate_cookbook.py recipes_final.json --title "Les Recettes de Maman"
"""

import sys
import os
import json
import argparse
from html import escape

# =============================================================================
# HTML TEMPLATE
# =============================================================================

HTML_HEADER = """\
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        /* ======= GOOGLE FONTS ======= */
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Lora:ital,wght@0,400;0,600;1,400&family=Inter:wght@300;400;500&display=swap');

        /* ======= PAGE SETUP ======= */
        @page {{
            size: A4;
            margin: 2.5cm 2cm;
            @bottom-right {{
                content: counter(page);
                font-family: 'Inter', sans-serif;
                font-size: 9pt;
                color: #999;
            }}
        }}

        :root {{
            --color-primary: #8B0000;
            --color-accent: #D4A574;
            --color-warm: #F5E6D3;
            --color-text: #2c2c2c;
            --color-muted: #666;
            --color-light: #f9f5f0;
            --color-border: #e8ddd0;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Lora', Georgia, 'Times New Roman', serif;
            font-size: 11pt;
            line-height: 1.7;
            color: var(--color-text);
            background-color: #fff;
        }}

        /* ======= COVER PAGE ======= */
        .cover {{
            height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            background: linear-gradient(135deg, var(--color-light) 0%, #fff 50%, var(--color-warm) 100%);
            page-break-after: always;
            position: relative;
            overflow: hidden;
        }}

        .cover::before {{
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 350px;
            height: 350px;
            border: 3px solid var(--color-accent);
            border-radius: 50%;
            opacity: 0.15;
        }}

        .cover::after {{
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 450px;
            height: 450px;
            border: 1px solid var(--color-accent);
            border-radius: 50%;
            opacity: 0.08;
        }}

        .cover h1 {{
            font-family: 'Playfair Display', Georgia, serif;
            font-size: 42pt;
            font-weight: 700;
            color: var(--color-primary);
            letter-spacing: 0.05em;
            margin-bottom: 0.2em;
            position: relative;
            z-index: 1;
        }}

        .cover .subtitle {{
            font-family: 'Playfair Display', Georgia, serif;
            font-style: italic;
            font-size: 16pt;
            color: var(--color-muted);
            margin-bottom: 2em;
            position: relative;
            z-index: 1;
        }}

        .cover .ornament {{
            font-size: 24pt;
            color: var(--color-accent);
            letter-spacing: 0.5em;
            position: relative;
            z-index: 1;
        }}

        .cover .recipe-count {{
            font-family: 'Inter', sans-serif;
            font-size: 10pt;
            color: var(--color-muted);
            margin-top: 2em;
            position: relative;
            z-index: 1;
        }}

        /* ======= TABLE OF CONTENTS ======= */
        .toc {{
            page-break-after: always;
            padding: 4em 3em;
        }}

        .toc h2 {{
            font-family: 'Playfair Display', Georgia, serif;
            font-size: 24pt;
            color: var(--color-primary);
            text-align: center;
            margin-bottom: 1.5em;
            letter-spacing: 0.05em;
        }}

        .toc-columns {{
            columns: 2;
            column-gap: 3em;
        }}

        .toc-entry {{
            display: flex;
            align-items: baseline;
            margin-bottom: 0.3em;
            font-size: 10pt;
            break-inside: avoid;
        }}

        .toc-number {{
            font-family: 'Inter', sans-serif;
            color: var(--color-accent);
            font-weight: 500;
            min-width: 2em;
            font-size: 9pt;
        }}

        .toc-title {{
            flex: 1;
            border-bottom: 1px dotted var(--color-border);
            padding-right: 0.5em;
        }}

        /* ======= RECIPE CARDS ======= */
        .recipe {{
            page-break-inside: avoid;
            margin-bottom: 3em;
            padding-bottom: 2em;
            border-bottom: 1px solid var(--color-border);
        }}

        .recipe:last-child {{
            border-bottom: none;
        }}

        .recipe-header {{
            margin-bottom: 1.5em;
            position: relative;
        }}

        .recipe-number {{
            font-family: 'Inter', sans-serif;
            font-size: 9pt;
            font-weight: 500;
            color: var(--color-accent);
            text-transform: uppercase;
            letter-spacing: 0.15em;
            margin-bottom: 0.3em;
        }}

        .recipe-title {{
            font-family: 'Playfair Display', Georgia, serif;
            font-size: 18pt;
            font-weight: 700;
            color: var(--color-primary);
            line-height: 1.3;
            margin-bottom: 0.2em;
        }}

        .recipe-divider {{
            width: 60px;
            height: 3px;
            background: linear-gradient(90deg, var(--color-accent), transparent);
            border: none;
            margin-top: 0.5em;
        }}

        /* ======= INGREDIENTS ======= */
        .ingredients {{
            background: var(--color-light);
            border-left: 4px solid var(--color-accent);
            padding: 1.2em 1.5em;
            margin-bottom: 1.5em;
            border-radius: 0 8px 8px 0;
        }}

        .section-label {{
            font-family: 'Inter', sans-serif;
            font-size: 9pt;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            color: var(--color-accent);
            margin-bottom: 0.7em;
        }}

        .ingredients ul {{
            list-style: none;
            padding: 0;
            columns: 2;
            column-gap: 2em;
        }}

        .ingredients li {{
            padding: 0.15em 0;
            padding-left: 1.2em;
            position: relative;
            font-size: 10.5pt;
            break-inside: avoid;
        }}

        .ingredients li::before {{
            content: '•';
            position: absolute;
            left: 0;
            color: var(--color-accent);
            font-weight: bold;
        }}

        /* ======= PREPARATION ======= */
        .preparation {{
            margin-bottom: 1.5em;
        }}

        .preparation ol {{
            padding: 0;
            list-style: none;
            counter-reset: steps;
        }}

        .preparation li {{
            counter-increment: steps;
            padding: 0.5em 0 0.5em 2.8em;
            position: relative;
            font-size: 10.5pt;
            line-height: 1.6;
        }}

        .preparation li::before {{
            content: counter(steps);
            position: absolute;
            left: 0;
            top: 0.45em;
            width: 2em;
            height: 2em;
            background: var(--color-primary);
            color: #fff;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'Inter', sans-serif;
            font-size: 9pt;
            font-weight: 500;
        }}

        /* ======= NOTES ======= */
        .notes {{
            background: #fff;
            border: 1px dashed var(--color-accent);
            border-radius: 8px;
            padding: 1em 1.5em;
            font-style: italic;
            font-size: 10pt;
            color: var(--color-muted);
        }}

        .notes::before {{
            content: '💡 ';
        }}

        /* ======= PRINT STYLES ======= */
        @media print {{
            body {{
                background: #fff;
            }}

            .cover {{
                height: auto;
                min-height: 90vh;
                background: #fff !important;
            }}

            .recipe {{
                page-break-inside: avoid;
            }}

            .ingredients {{
                background: #f9f5f0 !important;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}

            .preparation li::before {{
                background: var(--color-primary) !important;
                color: #fff !important;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
        }}

        /* ======= CONTENT CONTAINER ======= */
        .content {{
            max-width: 750px;
            margin: 0 auto;
            padding: 2em 3em;
        }}
    </style>
</head>
<body>
"""

HTML_FOOTER = """\
</body>
</html>
"""


# =============================================================================
# GENERATOR
# =============================================================================

def generate_cover(title: str, subtitle: str, count: int) -> str:
    """Generate the cover page HTML."""
    return f"""
    <div class="cover">
        <div class="ornament">✦ ✦ ✦</div>
        <h1>{escape(title)}</h1>
        <div class="subtitle">{escape(subtitle)}</div>
        <div class="ornament">✦ ✦ ✦</div>
        <div class="recipe-count">{count} recettes</div>
    </div>
    """


def generate_toc(recipes: list[dict]) -> str:
    """Generate the table of contents."""
    entries = []
    for r in recipes:
        num = r.get("numero", "")
        titre = escape(r.get("titre", "Sans titre"))
        entries.append(f"""
            <div class="toc-entry">
                <span class="toc-number">{num}.</span>
                <span class="toc-title">{titre}</span>
            </div>
        """)

    return f"""
    <div class="toc">
        <h2>Table des Matières</h2>
        <div class="toc-columns">
            {''.join(entries)}
        </div>
    </div>
    """


def generate_recipe(recipe: dict) -> str:
    """Generate the HTML for a single recipe card."""
    numero = recipe.get("numero", "")
    titre = escape(recipe.get("titre", "Sans titre"))
    ingredients = recipe.get("ingredients", [])
    preparation = recipe.get("preparation", [])
    notes = recipe.get("notes")

    # Ingredients list
    ing_items = "\n".join(f"            <li>{escape(i)}</li>" for i in ingredients)

    # Preparation steps
    prep_items = "\n".join(f"            <li>{escape(s)}</li>" for s in preparation)

    # Notes section (only if present)
    notes_html = ""
    if notes and notes.strip():
        notes_html = f"""
        <div class="notes">
            {escape(notes)}
        </div>
        """

    return f"""
    <div class="recipe" id="recipe-{numero}">
        <div class="recipe-header">
            <div class="recipe-number">Recette nº {numero}</div>
            <div class="recipe-title">{titre}</div>
            <hr class="recipe-divider">
        </div>

        <div class="ingredients">
            <div class="section-label">Ingrédients</div>
            <ul>
{ing_items}
            </ul>
        </div>

        <div class="preparation">
            <div class="section-label">Préparation</div>
            <ol>
{prep_items}
            </ol>
        </div>

        {notes_html}
    </div>
    """


def generate_cookbook(recipes: list[dict], title: str, subtitle: str) -> str:
    """Generate the full HTML cookbook."""
    parts = []
    parts.append(HTML_HEADER.format(title=escape(title)))
    parts.append(generate_cover(title, subtitle, len(recipes)))
    parts.append(generate_toc(recipes))
    parts.append('<div class="content">')

    for recipe in recipes:
        parts.append(generate_recipe(recipe))

    parts.append('</div>')
    parts.append(HTML_FOOTER)

    return "\n".join(parts)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate a beautifully formatted HTML cookbook from a recipes JSON file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python generate_cookbook.py recipes_final.json
  python generate_cookbook.py recipes_final.json --output mon_cahier.html
  python generate_cookbook.py recipes_final.json --title "Les Recettes de Grand-Maman"
"""
    )
    parser.add_argument("input_file", help="Path to the recipes JSON file")
    parser.add_argument("--output", "-o", default="cookbook.html",
                        help="Output HTML file path (default: cookbook.html)")
    parser.add_argument("--title", default="Cahier de Recettes",
                        help='Book title (default: "Cahier de Recettes")')
    parser.add_argument("--subtitle", default="Recettes de famille, transmises avec amour",
                        help="Subtitle for the cover page")

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: File not found: {args.input_file}")
        sys.exit(1)

    # Load recipes
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    recipes = data.get("recettes", [])

    print(f"\n{'='*60}")
    print(f"Cookbook Generator")
    print(f"{'='*60}")
    print(f"Input: {args.input_file}")
    print(f"Recipes: {len(recipes)}")
    print(f"Title: {args.title}")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")

    # Generate
    html = generate_cookbook(recipes, args.title, args.subtitle)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  ✅ Cookbook generated: {args.output}")
    print(f"     Open in a browser to view, edit, or print to PDF.")
    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
