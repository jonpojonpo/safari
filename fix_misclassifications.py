#!/usr/bin/env python3
"""Fix misclassified photos and rebuild the logbook."""

import json, base64, asyncio, time
from pathlib import Path
from io import BytesIO
from PIL import Image
import anthropic

OUT_DIR = Path.home() / "safari" / "data"
CLASSIFICATIONS_FILE = OUT_DIR / "classifications.json"
ANIMAL_INFO_FILE = OUT_DIR / "animal_info.json"
HEROES_DIR = OUT_DIR / "heroes"
THUMBS_DIR = OUT_DIR / "thumbs"
PHOTOS_DIRS = [
    Path.home() / "Movies",
    Path.home() / "safari" / "data" / "new-raw-photos",
    Path.home() / "safari" / "data" / "new-raw-photo2",
    Path.home() / "safari" / "data" / "new-raw-photo3",
]

client = anthropic.Anthropic()

def find_photo(filename):
    for d in PHOTOS_DIRS:
        p = d / filename
        if p.exists():
            return p
    return None

def encode_image(path, max_width=1200):
    with Image.open(path) as img:
        img = img.convert("RGB")
        w, h = img.size
        if w > max_width:
            scale = max_width / w
            img = img.resize((max_width, int(h * scale)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, "JPEG", quality=85)
        return base64.standard_b64encode(buf.getvalue()).decode()

def load_json(path, default=None):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return default

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# Animals impossible or very unlikely on an African safari
RECLASSIFY_PROMPT = """This photo was taken on an AFRICAN SAFARI. The previous classification was '{wrong_animal}' which is incorrect for Africa.

The confirmed species present on this safari are: Lion, Cheetah, African Elephant, Zebra, Impala, White Rhino, Hippo, Vervet Monkey, Greater Kudu, Southern Yellow-billed Hornbill, Warthog, Crocodile, Puff Adder, various vultures and small birds.

There are NO leopards, tigers, chitals, tufted titmice, gopher snakes or turkey vultures on this African safari.

Please identify the CORRECT African species. Return ONLY a valid JSON object:
{{
  "animal": "correct common name",
  "scientific_name": "correct scientific binomial",
  "animal_count": <integer>,
  "quality_score": <1-10>,
  "technical_score": <1-10>,
  "behaviour": "what the animal is doing",
  "notes": "one sentence on photo strengths or weaknesses"
}}"""

SUSPECTS = {
    "Leopard": "Leopard — confirmed not present on this safari, likely Cheetah or Lion",
    "Tiger": "Tiger — impossible in Africa",
    "Chital": "Chital — Indian deer, not present in Africa",
    "Tufted Titmouse": "Tufted Titmouse — North American bird",
    "Turkey Vulture": "Turkey Vulture — American species",
    "Gopher Snake": "Gopher Snake — North American species",
    "Bird": "Bird — too vague, needs proper identification",
}

def reclassify_all():
    classifications = load_json(CLASSIFICATIONS_FILE, {})

    to_fix = {
        name: data for name, data in classifications.items()
        if data.get("animal") in SUSPECTS
    }

    print(f"Reclassifying {len(to_fix)} photos...")

    for filename, data in to_fix.items():
        wrong = data["animal"]
        photo_path = find_photo(filename)
        if not photo_path:
            print(f"  ✗ {filename}: source not found")
            continue

        prompt = RECLASSIFY_PROMPT.format(wrong_animal=SUSPECTS[wrong])
        img_b64 = encode_image(photo_path)

        try:
            resp = client.messages.create(
                model="claude-opus-4-8",
                max_tokens=512,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                        {"type": "text", "text": prompt}
                    ]
                }]
            )
            text = resp.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            new_data = json.loads(text)
            new_data["file"] = filename
            classifications[filename] = new_data
            print(f"  ✓ {filename}: {wrong} → {new_data['animal']} (score {new_data.get('quality_score')})")
            time.sleep(0.3)
        except Exception as e:
            print(f"  ✗ {filename}: {e}")

    save_json(CLASSIFICATIONS_FILE, classifications)
    print(f"\nSaved updated classifications.")
    return classifications


def rebuild_logbook(classifications):
    """Re-run stages 2-5 from the main pipeline."""
    import sys
    sys.path.insert(0, str(Path.home() / "safari"))

    # Inline the select + web assets + animal info + HTML logic
    from pipeline import (
        select_best_photos, generate_web_assets, generate_animal_info,
        build_html, load_json as lpj, save_json as spj, ANIMAL_INFO_FILE,
        TOP_N_PER_ANIMAL
    )

    print("\nRebuilding logbook...")
    selected = select_best_photos(classifications)
    generate_web_assets(selected)

    # Refresh animal info for new/changed animals
    animal_info = generate_animal_info(selected)

    html = build_html(selected, animal_info)
    out_path = Path.home() / "safari" / "index.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"\nDone! Open: file://{out_path}")


if __name__ == "__main__":
    classifications = reclassify_all()
    rebuild_logbook(classifications)
