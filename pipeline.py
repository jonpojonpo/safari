#!/usr/bin/env python3
"""
Safari Photo Logbook Pipeline
Classifies, rates, and organises safari photos into a National Geographic-style web logbook.
"""

import os
import json
import base64
import asyncio
import time
import html
import re
from pathlib import Path
from io import BytesIO
from PIL import Image, ImageOps
import anthropic

# ── Config ────────────────────────────────────────────────────────────────────
PHOTOS_DIRS = [
    Path.home() / "Movies",
    Path.home() / "safari" / "data" / "new-raw-photos",
    Path.home() / "safari" / "data" / "new-raw-photo2",
    Path.home() / "safari" / "data" / "new-raw-photo3",
    Path.home() / "safari" / "data" / "new-raw-photo4",
]
OUT_DIR = Path.home() / "safari" / "data"
CLASSIFICATIONS_FILE = OUT_DIR / "classifications.json"
ANIMAL_INFO_FILE = OUT_DIR / "animal_info.json"

WEB_PHOTOS_DIR = OUT_DIR / "web_photos"
HEROES_DIR = OUT_DIR / "heroes"
THUMBS_DIR = OUT_DIR / "thumbs"
DETAILS_DIR = OUT_DIR / "details"
CLASSIFY_CACHE_DIR = OUT_DIR / "classify_cache"

API_CLASSIFY_WIDTH = int(os.getenv("SAFARI_CLASSIFY_WIDTH", "768"))  # px — for Claude vision analysis
HERO_WIDTH = 1920           # px — full-bleed hero images
THUMB_WIDTH = 600           # px — gallery thumbnails
DETAIL_WIDTH = 900          # px — tighter subject/detail crops

MAX_CONCURRENT = int(os.getenv("SAFARI_MAX_CONCURRENT", "2"))  # parallel API calls
TOP_N_PER_ANIMAL = 5        # best photos per animal shown in logbook
PHOTO_GLOBS = ("*.JPG", "*.JPEG", "*.jpg", "*.jpeg", "*.PNG", "*.png")

client = anthropic.Anthropic()

for directory in (WEB_PHOTOS_DIR, HEROES_DIR, THUMBS_DIR, DETAILS_DIR, CLASSIFY_CACHE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def resize_image_bytes(path: Path, max_width: int, quality: int = 85) -> bytes:
    """Resize a JPEG to max_width, return JPEG bytes."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        w, h = img.size
        if w > max_width:
            scale = max_width / w
            img = img.resize((max_width, int(h * scale)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, "JPEG", quality=quality, optimize=True)
        return buf.getvalue()


def encode_image(path: Path, max_width: int) -> str:
    """Return base64-encoded optimized JPEG, using a disk cache for retries/resumes."""
    cache_key = f"{path.stem}_{path.stat().st_size}_{path.stat().st_mtime_ns}_{max_width}.jpg"
    cache_path = CLASSIFY_CACHE_DIR / cache_key
    if not cache_path.exists():
        cache_path.write_bytes(resize_image_bytes(path, max_width, quality=76))
    return base64.standard_b64encode(cache_path.read_bytes()).decode()


def load_json(path: Path, default):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return default


def save_json(path: Path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def safe_text(value) -> str:
    return html.escape(str(value or ""), quote=True)


def js_text(value) -> str:
    return json.dumps(str(value or ""))


def display_note(photo_data: dict) -> str:
    """Convert blunt internal critique into warm public-facing photo copy."""
    animal = photo_data.get("animal", "wildlife")
    behaviour = photo_data.get("behaviour") or "a quiet field moment"
    score = int(photo_data.get("quality_score", 0) or 0)
    if score >= 8:
        return f"A standout frame: the composition brings out the character of the {animal} and the atmosphere of the encounter."
    if score >= 6:
        return f"A lovely field-journal image, catching the {animal} in {behaviour} with a clear sense of place."
    return f"An honest safari moment: even in challenging light and movement, the frame preserves the thrill of spotting the {animal}."


def public_behaviour(photo_data: dict) -> str:
    behaviour = str(photo_data.get("behaviour") or "").strip()
    if not behaviour or behaviour.upper() == "N/A":
        return "observed in the landscape"
    return behaviour


def photo_sort_key(path: Path):
    match = re.search(r"(\d+)", path.stem)
    return (int(match.group(1)) if match else 10**9, path.name)


def iter_photo_files(directory: Path):
    for pattern in PHOTO_GLOBS:
        yield from directory.glob(pattern)


def centre_crop_box(img: Image.Image, target_ratio: float) -> tuple[int, int, int, int]:
    """Crop to the target aspect ratio about the centre — predictable, never off-subject.

    A previous saliency-based 'smart' crop chased background detail (bushes, grass)
    and shoved the subject against an edge, so framing is now a plain centre crop.
    """
    w, h = img.size
    if w / h > target_ratio:
        crop_w = int(h * target_ratio)
        crop_h = h
    else:
        crop_w = w
        crop_h = int(w / target_ratio)
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    return (left, top, left + crop_w, top + crop_h)


def save_smart_crop(src: Path, dest: Path, width: int, ratio: float, quality: int, zoom: float = 1.0):
    """Resize to a centre crop of the given aspect ratio. `zoom` is accepted for
    backwards-compatibility but no longer tightens the frame (no auto-cropping)."""
    with Image.open(src) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        crop = img.crop(centre_crop_box(img, ratio))
        if crop.width > width:
            height = int(crop.height * (width / crop.width))
            crop = crop.resize((width, height), Image.LANCZOS)
        crop.save(dest, "JPEG", quality=quality, optimize=True, progressive=True)


# ── Stage 1: Classify & Rate ──────────────────────────────────────────────────
CLASSIFY_PROMPT = """You are analysing a wildlife photograph taken on safari.

Return ONLY a valid JSON object with these exact keys:
{
  "animal": "common name of the primary animal (e.g. 'African Elephant', 'Lion', 'Giraffe', 'Zebra', 'Cheetah', 'Leopard', 'Cape Buffalo', 'Hippo', 'Warthog', 'Impala', 'Wildebeest', 'Baboon', 'Vervet Monkey', 'Crocodile', 'Ostrich', 'Marabou Stork', 'African Wild Dog', 'Hyena', 'Rhino', 'No Animal' if none visible)",
  "scientific_name": "scientific binomial name",
  "animal_count": <integer number of animals visible>,
  "quality_score": <integer 1-10 where 10=exceptional; consider: sharp focus, good exposure, compelling composition, animal behaviour, background clarity>,
  "technical_score": <integer 1-10 for pure technical quality: focus sharpness, exposure, noise>,
  "behaviour": "brief description of what the animal is doing (e.g. 'grazing', 'running', 'resting', 'drinking', 'nursing calf')",
  "notes": "one encouraging sentence about the strongest wildlife, composition, atmosphere, or field-observation quality in the image"
}

Be precise with the numeric scores, but keep notes constructive and generous.
Return ONLY the JSON object, no other text."""


RECLASSIFY_PROMPT = """This photo was taken on an African safari. A previous classification labelled this as a 'Gopher Snake' which is incorrect — that is a North American species.

Please identify the correct African species. Return ONLY a valid JSON object with the same keys as before:
{
  "animal": "correct common name of this African snake species",
  "scientific_name": "correct scientific binomial",
  "animal_count": 1,
  "quality_score": <1-10>,
  "technical_score": <1-10>,
  "behaviour": "what the snake is doing",
  "notes": "one sentence on the photo's strengths or weaknesses"
}"""


async def classify_photo(sem: asyncio.Semaphore, photo: Path, results: dict) -> dict:
    """Classify a single photo with Claude vision."""
    name = photo.name
    if name in results:
        return results[name]  # already done

    async with sem:
        try:
            img_b64 = encode_image(photo, API_CLASSIFY_WIDTH)
            response = await asyncio.to_thread(
                client.messages.create,
                model="claude-opus-4-8",
                max_tokens=512,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_b64,
                            }
                        },
                        {"type": "text", "text": CLASSIFY_PROMPT}
                    ]
                }],
            )
            text = response.content[0].text.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)
            data["file"] = name
            print(f"  ✓ {name}: {data.get('animal','?')} score={data.get('quality_score','?')}")
            return data
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            return {"file": name, "animal": "Unknown", "quality_score": 0, "error": str(e)}


def find_photo(filename: str) -> Path | None:
    """Locate a photo file across all source directories."""
    for d in PHOTOS_DIRS:
        p = d / filename
        if p.exists():
            return p
    return None


async def run_classifications():
    """Classify all photos, resuming from saved progress."""
    results = load_json(CLASSIFICATIONS_FILE, {})

    # Collect all photos from all source directories (deduplicate by filename)
    seen = set()
    photos = []
    for d in PHOTOS_DIRS:
        for p in sorted(iter_photo_files(d), key=photo_sort_key):
            if p.name not in seen:
                seen.add(p.name)
                photos.append(p)

    # ── Fix misclassified snake(s) ──
    snake_fixes = [name for name, d in results.items() if d.get("animal") == "Gopher Snake"]
    if snake_fixes:
        print(f"\n  Fixing {len(snake_fixes)} misclassified snake photo(s)...")
        for name in snake_fixes:
            photo_path = find_photo(name)
            if not photo_path:
                continue
            try:
                img_b64 = encode_image(photo_path, API_CLASSIFY_WIDTH)
                resp = await asyncio.to_thread(
                    client.messages.create,
                    model="claude-opus-4-8",
                    max_tokens=512,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                            {"type": "text", "text": RECLASSIFY_PROMPT}
                        ]
                    }],
                )
                text = resp.content[0].text.strip()
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                data = json.loads(text)
                data["file"] = name
                results[name] = data
                print(f"  ✓ Reclassified {name}: {data.get('animal')} (was Gopher Snake)")
            except Exception as e:
                print(f"  ✗ Reclassify failed for {name}: {e}")
        save_json(CLASSIFICATIONS_FILE, results)

    already_done = len([p for p in photos if p.name in results])
    todo = [p for p in photos if p.name not in results]
    print(f"\n{'─'*60}")
    print(f"STAGE 1: Classify & Rate  ({already_done} done, {len(todo)} remaining)")
    print(f"{'─'*60}")

    sem = asyncio.Semaphore(MAX_CONCURRENT)

    async def process(photo):
        result = await classify_photo(sem, photo, results)
        results[photo.name] = result
        # Save progress every 10 photos
        if len(results) % 10 == 0:
            save_json(CLASSIFICATIONS_FILE, results)

    # Run in batches to be gentle on rate limits
    batch_size = 20
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i+batch_size]
        await asyncio.gather(*[process(p) for p in batch])
        save_json(CLASSIFICATIONS_FILE, results)
        print(f"  Progress: {min(already_done + i + batch_size, len(photos))}/{len(photos)}")
        if i + batch_size < len(todo):
            await asyncio.sleep(1)  # brief pause between batches

    save_json(CLASSIFICATIONS_FILE, results)
    print(f"  Classifications saved → {CLASSIFICATIONS_FILE}")
    return results


# ── Stage 2: Select Best Photos ───────────────────────────────────────────────
def select_best_photos(classifications: dict, top_n: int = TOP_N_PER_ANIMAL) -> dict:
    """Group by animal, return top N per animal by quality score."""
    print(f"\n{'─'*60}")
    print(f"STAGE 2: Select Top {top_n} Per Animal")
    print(f"{'─'*60}")

    by_animal = {}
    for name, data in classifications.items():
        animal = data.get("animal", "Unknown")
        if animal in ("Unknown", "No Animal", "None"):
            continue
        if animal not in by_animal:
            by_animal[animal] = []
        by_animal[animal].append(data)

    selected = {}
    for animal, photos in sorted(by_animal.items()):
        photos_sorted = sorted(photos, key=lambda x: x.get("quality_score", 0), reverse=True)
        selected[animal] = photos_sorted[:top_n]
        scores = [p.get("quality_score", 0) for p in photos_sorted[:top_n]]
        print(f"  {animal}: {len(photos)} photos → top {min(top_n, len(photos))} (scores: {scores})")

    return selected


# ── Stage 3: Generate Web Assets ─────────────────────────────────────────────
def generate_web_assets(selected: dict):
    """Create smart-cropped hero, detail, and thumbnail images for selected photos."""
    print(f"\n{'─'*60}")
    print(f"STAGE 3: Generate Web Assets")
    print(f"{'─'*60}")

    for animal, photos in selected.items():
        for i, photo_data in enumerate(photos):
            fname = photo_data["file"]
            src = find_photo(fname)
            if not src:
                print(f"  ✗ Source not found: {fname}")
                continue
            stem = Path(fname).stem

            # Hero (first/best photo per animal gets extra-large treatment)
            hero_path = HEROES_DIR / f"{stem}_hero.jpg"
            if not hero_path.exists():
                save_smart_crop(src, hero_path, HERO_WIDTH, ratio=16/9, quality=88, zoom=1.0)
                print(f"  hero  → {hero_path.name}")

            # Detail crop gives shy or distant subjects a better moment in the frame.
            detail_path = DETAILS_DIR / f"{stem}_detail.jpg"
            if not detail_path.exists():
                save_smart_crop(src, detail_path, DETAIL_WIDTH, ratio=4/3, quality=86, zoom=1.35)
                print(f"  detail→ {detail_path.name}")

            # Thumbnail
            thumb_path = THUMBS_DIR / f"{stem}_thumb.jpg"
            if not thumb_path.exists():
                save_smart_crop(src, thumb_path, THUMB_WIDTH, ratio=3/2, quality=82, zoom=1.15)
                print(f"  thumb → {thumb_path.name}")


# ── Stage 4: Generate Animal Background Info ──────────────────────────────────
ANIMAL_INFO_PROMPT = """You are a National Geographic wildlife writer.
Write rich, compelling content about the {animal} ({scientific_name}) for a safari logbook.

Return ONLY a JSON object:
{{
  "tagline": "a dramatic, evocative 8-10 word tagline for this animal",
  "intro": "a gripping 2-3 sentence introduction in National Geographic style — vivid and cinematic",
  "habitat": "one paragraph about habitat and range in Africa",
  "behaviour": "one paragraph about fascinating behaviour, social structure, hunting/feeding",
  "conservation": "one sentence on IUCN status and main threats",
  "fast_facts": [
    "Fact 1 (e.g. weight/size)",
    "Fact 2 (e.g. speed or lifespan)",
    "Fact 3 (e.g. unique adaptation or record)",
    "Fact 4 (e.g. diet or prey)",
    "Fact 5 (e.g. social behaviour or territory)"
  ]
}}

Be scientifically accurate but write with passion and drama. Keep the tone warm and admiring of the photographer's fieldcraft: curious, patient, observant, and learning by looking."""


def generate_animal_info(selected: dict) -> dict:
    """Generate rich background content for each animal."""
    print(f"\n{'─'*60}")
    print(f"STAGE 4: Generate Animal Background Content")
    print(f"{'─'*60}")

    animal_info = load_json(ANIMAL_INFO_FILE, {})

    for animal, photos in selected.items():
        if animal in animal_info:
            print(f"  ✓ {animal} (cached)")
            continue

        sci_name = photos[0].get("scientific_name", "")
        prompt = ANIMAL_INFO_PROMPT.format(animal=animal, scientific_name=sci_name)

        try:
            response = client.messages.create(
                model="claude-opus-4-8",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)
            animal_info[animal] = data
            print(f"  ✓ {animal}: \"{data.get('tagline', '')}\"")
            save_json(ANIMAL_INFO_FILE, animal_info)
            time.sleep(0.3)
        except Exception as e:
            print(f"  ✗ {animal}: {e}")

    return animal_info


# ── Stage 5: Build HTML Logbook ───────────────────────────────────────────────
def build_html(selected: dict, animal_info: dict) -> str:
    """Generate the complete National Geographic-style HTML logbook."""
    print(f"\n{'─'*60}")
    print(f"STAGE 5: Build HTML Logbook")
    print(f"{'─'*60}")

    # The Big Five lead the journal in their traditional order; the rest follow A–Z.
    BIG_FIVE = ["Lion", "Leopard", "African Elephant", "Cape Buffalo", "White Rhino"]
    big_five_present = [a for a in BIG_FIVE if a in selected]
    big_five_set = set(big_five_present)
    others = sorted(a for a in selected if a not in big_five_set)
    animals = big_five_present + others

    def anchor_of(a: str) -> str:
        return safe_text(a.replace(" ", "-").lower())

    def hero_src_of(a: str) -> str:
        return f"data/heroes/{Path(selected[a][0]['file']).stem}_hero.jpg"

    # ── Nav items (Big Five flagged with a star)
    nav_items = "\n".join(
        f'<li><a href="#animal-{anchor_of(a)}" title="{safe_text(a)}" '
        f'class="nav-link{" nav-big5" if a in big_five_set else ""}">'
        f'{"★ " if a in big_five_set else ""}{safe_text(a)}</a></li>'
        for a in animals
    )

    # ── Big Five showcase band (only when the full set is present)
    big_five_band = ""
    if len(big_five_present) == 5:
        tiles = "\n".join(
            f'''<a class="bf-tile" href="#animal-{anchor_of(a)}" style="background-image:url('{hero_src_of(a)}')">
          <span class="bf-tile-overlay"></span>
          <span class="bf-tile-name">{safe_text(a)}</span>
        </a>'''
            for a in big_five_present
        )
        big_five_band = f'''
<section class="bigfive-band">
  <div class="bigfive-head">
    <div class="bigfive-eyebrow">The Big Five — Complete</div>
    <h2 class="bigfive-title">Africa's Legendary Five</h2>
    <p class="bigfive-sub">Lion, leopard, elephant, buffalo and rhino — the full set, photographed in the field.</p>
  </div>
  <div class="bigfive-grid">
    {tiles}
  </div>
</section>'''

    # ── Animal sections
    def make_animal_section(animal: str, photos: list, info: dict) -> str:
        anchor = safe_text(animal.replace(" ", "-").lower())
        sci_name = safe_text(photos[0].get("scientific_name", ""))
        best = photos[0]
        best_stem = Path(best["file"]).stem
        hero_src = f"data/heroes/{best_stem}_hero.jpg"
        detail_src = f"data/details/{best_stem}_detail.jpg"

        info_data = info.get(animal, {})
        tagline = safe_text(info_data.get("tagline", animal))
        intro = safe_text(info_data.get("intro", ""))
        field_note = safe_text(info_data.get("field_note", ""))
        habitat = safe_text(info_data.get("habitat", ""))
        behaviour = safe_text(info_data.get("behaviour", ""))
        conservation = safe_text(info_data.get("conservation", ""))
        fast_facts = info_data.get("fast_facts", [])

        fast_facts_html = "\n".join(
            f'<div class="fact-item"><span class="fact-bullet">◆</span>{safe_text(fact)}</div>'
            for fact in fast_facts
        )

        # Gallery thumbnails (all selected photos)
        gallery_items = []
        for p in photos:
            stem = Path(p["file"]).stem
            score = int(p.get("quality_score", 0) or 0)
            behaviour_str = public_behaviour(p)
            notes = display_note(p)
            thumb_src = f"data/thumbs/{stem}_thumb.jpg"
            detail_src_g = f"data/details/{stem}_detail.jpg"
            hero_src_g = f"data/heroes/{stem}_hero.jpg"
            stars = "★" * score + "☆" * (10 - score)
            gallery_items.append(f"""
        <div class="gallery-item" onclick='openLightbox({js_text(hero_src_g)}, {js_text(detail_src_g)}, {js_text(behaviour_str)}, {js_text(notes)})'>
          <img src="{thumb_src}" alt="{safe_text(animal)}" loading="lazy">
          <div class="gallery-caption">
            <div class="gallery-score">{stars[:5]}</div>
            <div class="gallery-behaviour">{safe_text(behaviour_str)}</div>
          </div>
        </div>""")

        gallery_html = "\n".join(gallery_items)

        total_photos = len(photos)
        best_score = int(best.get("quality_score", 0) or 0)
        best_behaviour = public_behaviour(best)
        best_note = display_note(best)

        return f"""
  <section class="animal-section" id="animal-{anchor}">
    <div class="hero-wrap">
      <img class="hero-img" src="{hero_src}" alt="{safe_text(animal)}" loading="lazy">
      <div class="hero-overlay">
        <div class="hero-overlay-inner">
          {'<div class="hero-big5">★ Big Five</div>' if animal in big_five_set else ''}
          <div class="hero-sci">{sci_name}</div>
          <h2 class="hero-name">{safe_text(animal)}</h2>
          <p class="hero-tagline">{tagline}</p>
          <div class="hero-badge">
            <span class="hero-score-label">Field favourite</span>
            <span class="hero-score">{"★" * best_score}</span>
          </div>
        </div>
      </div>
    </div>

    <div class="content-wrap">
      <div class="content-main">
        <p class="intro-text">{intro}</p>
        {f'<div class="field-note"><span class="field-note-label">◈ From the Field — The Find</span><p>{field_note}</p></div>' if field_note else ''}
        <figure class="detail-feature" onclick='openLightbox({js_text(hero_src)}, {js_text(detail_src)}, {js_text(best_behaviour)}, {js_text(best_note)})'>
          <img src="{detail_src}" alt="Detail crop of {safe_text(animal)}" loading="lazy">
          <figcaption>
            <span>Photographer's detail</span>
            {safe_text(best_note)}
          </figcaption>
        </figure>

        <div class="two-col">
          <div class="col-text">
            <h3 class="section-heading">Habitat &amp; Range</h3>
            <p>{habitat}</p>
            <h3 class="section-heading">Behaviour</h3>
            <p>{behaviour}</p>
            <p class="conservation-note">🔴 {conservation}</p>
          </div>
          <div class="col-facts">
            <h3 class="section-heading">Fast Facts</h3>
            <div class="facts-list">
              {fast_facts_html}
            </div>
            <div class="encounter-stats">
              <h3 class="section-heading">Encounter</h3>
              <div class="stat-row"><span class="stat-label">Photos taken</span><span class="stat-value">{total_photos} selected</span></div>
              <div class="stat-row"><span class="stat-label">Observed moment</span><span class="stat-value">{safe_text(best_behaviour)}</span></div>
              <div class="stat-row"><span class="stat-label">Photo pick</span><span class="stat-value">{"★" * best_score} Field favourite</span></div>
            </div>
          </div>
        </div>

        <h3 class="section-heading gallery-heading">Gallery — Top Shots</h3>
        <div class="photo-gallery">
          {gallery_html}
        </div>
      </div>
    </div>
  </section>"""

    all_sections = "\n".join(
        make_animal_section(a, selected[a], animal_info)
        for a in animals
    )

    total_photos_analysed = sum(len(v) for v in selected.values())
    all_animals_count = len(animals)
    all_source_photo_count = len({p.name for d in PHOTOS_DIRS for p in iter_photo_files(d)})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Safari Field Journal</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,600;0,700;1,400&family=Source+Sans+3:wght@300;400;600&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    :root {{
      --gold: #c9a227;
      --gold-light: #f0c040;
      --black: #080808;
      --dark: #111111;
      --panel: #181818;
      --border: #2a2a2a;
      --text: #d8d0c0;
      --text-dim: #7a7060;
      --serif: 'EB Garamond', Georgia, serif;
      --sans: 'Source Sans 3', system-ui, sans-serif;
    }}

    html {{ scroll-behavior: smooth; }}

    body {{
      background: var(--black);
      color: var(--text);
      font-family: var(--sans);
      font-size: 17px;
      line-height: 1.7;
    }}

    /* ── Cover ── */
    .cover {{
      height: 100vh;
      min-height: 600px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      text-align: center;
      background: linear-gradient(160deg, #0a0a00 0%, #1a0a00 50%, #000 100%);
      position: relative;
      overflow: hidden;
      padding: 2rem;
    }}
    .cover::before {{
      content: '';
      position: absolute;
      inset: 0;
      background: radial-gradient(ellipse at 50% 40%, rgba(201,162,39,0.08) 0%, transparent 70%);
    }}
    .cover-rule {{ width: 80px; height: 3px; background: var(--gold); margin: 0 auto 2rem; }}
    .cover-eyebrow {{
      font-family: var(--sans);
      font-weight: 600;
      font-size: 0.75rem;
      letter-spacing: 0.25em;
      text-transform: uppercase;
      color: var(--gold);
      margin-bottom: 1.5rem;
    }}
    .cover-title {{
      font-family: var(--serif);
      font-size: clamp(2.8rem, 8vw, 6rem);
      font-weight: 700;
      line-height: 1.05;
      color: #fff;
      margin-bottom: 1.5rem;
    }}
    .cover-subtitle {{
      font-family: var(--serif);
      font-style: italic;
      font-size: clamp(1.1rem, 2.5vw, 1.5rem);
      color: var(--text-dim);
      max-width: 500px;
      margin: 0 auto 3rem;
    }}
    .cover-stats {{
      display: flex;
      gap: 3rem;
      justify-content: center;
      flex-wrap: wrap;
    }}
    .cover-stat {{ text-align: center; }}
    .cover-stat-num {{
      font-family: var(--serif);
      font-size: 2.5rem;
      font-weight: 700;
      color: var(--gold);
      line-height: 1;
    }}
    .cover-stat-label {{
      font-size: 0.7rem;
      letter-spacing: 0.15em;
      text-transform: uppercase;
      color: var(--text-dim);
      margin-top: 0.3rem;
    }}
    .scroll-hint {{
      position: absolute;
      bottom: 2rem;
      left: 50%;
      transform: translateX(-50%);
      color: var(--text-dim);
      font-size: 0.75rem;
      letter-spacing: 0.2em;
      text-transform: uppercase;
      animation: bounce 2s infinite;
    }}
    @keyframes bounce {{
      0%, 100% {{ transform: translateX(-50%) translateY(0); }}
      50% {{ transform: translateX(-50%) translateY(6px); }}
    }}

    /* ── Side Nav ── */
    .side-nav {{
      position: fixed;
      left: 0;
      top: 50%;
      transform: translateY(-50%);
      z-index: 100;
      padding: 1rem 0;
      max-height: 90vh;
      overflow-y: auto;
    }}
    .side-nav ul {{ list-style: none; }}
    .side-nav .nav-link {{
      display: block;
      padding: 0.35rem 1.2rem;
      font-size: 0.7rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--text-dim);
      text-decoration: none;
      border-left: 2px solid transparent;
      transition: all 0.2s;
      white-space: nowrap;
      max-width: 170px;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .side-nav .nav-link:hover {{
      color: var(--gold);
      border-left-color: var(--gold);
    }}
    @media (max-width: 900px) {{ .side-nav {{ display: none; }} }}

    /* ── Animal Section ── */
    .animal-section {{
      margin-bottom: 0;
    }}

    /* ── Hero ── */
    .hero-wrap {{
      position: relative;
      height: 90vh;
      min-height: 500px;
      overflow: hidden;
    }}
    .hero-img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      object-position: center;
      display: block;
    }}
    .hero-overlay {{
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
      padding: 3rem 2rem;
      background: linear-gradient(transparent 0%, rgba(0,0,0,0.5) 30%, rgba(0,0,0,0.85) 100%);
    }}
    /* Hero title shares the body content's horizontal layout so their left
       edges line up exactly (same padding + centred max-width column). */
    .hero-overlay-inner {{
      max-width: 1000px;
      margin: 0 auto;
    }}
    @media (min-width: 1100px) {{
      .hero-overlay {{ padding-left: 160px; padding-right: 160px; }}
    }}
    .hero-sci {{
      font-family: var(--serif);
      font-style: italic;
      font-size: 1rem;
      color: var(--gold);
      margin-bottom: 0.4rem;
    }}
    .hero-name {{
      font-family: var(--serif);
      font-size: clamp(2.5rem, 6vw, 5rem);
      font-weight: 700;
      color: #fff;
      line-height: 1.1;
      margin-bottom: 0.6rem;
    }}
    .hero-tagline {{
      font-family: var(--serif);
      font-style: italic;
      font-size: clamp(1rem, 2vw, 1.35rem);
      color: rgba(255,255,255,0.8);
      max-width: 600px;
      margin-bottom: 1rem;
    }}
    .hero-badge {{
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      background: rgba(0,0,0,0.5);
      border: 1px solid var(--gold);
      padding: 0.4rem 1rem;
      border-radius: 2px;
    }}
    .hero-score-label {{
      font-size: 0.65rem;
      letter-spacing: 0.15em;
      text-transform: uppercase;
      color: var(--text-dim);
    }}
    .hero-score {{ color: var(--gold); font-size: 0.9rem; }}

    /* ── Content ── */
    .content-wrap {{
      background: var(--dark);
      padding: 4rem 2rem 5rem;
    }}
    .content-main {{
      max-width: 1000px;
      margin: 0 auto;
    }}
    /* On wide screens reserve room for the fixed side-nav, then centre the
       content within the remaining space so margins stay balanced. */
    @media (min-width: 1100px) {{
      .content-wrap {{ padding-left: 160px; padding-right: 160px; }}
    }}
    .intro-text {{
      font-family: var(--serif);
      font-size: clamp(1.1rem, 2vw, 1.35rem);
      line-height: 1.75;
      color: rgba(216, 208, 192, 0.95);
      margin-bottom: 3rem;
      padding-bottom: 2rem;
      border-bottom: 1px solid var(--border);
    }}

    .field-note {{
      margin: -1rem 0 2.5rem;
      padding: 1.4rem 1.6rem;
      background: linear-gradient(135deg, rgba(201,162,39,0.10), rgba(255,255,255,0.02));
      border-left: 3px solid var(--gold);
    }}
    .field-note-label {{
      display: block;
      font-family: var(--sans);
      font-size: 0.65rem;
      font-weight: 600;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--gold);
      margin-bottom: 0.7rem;
    }}
    .field-note p {{
      font-family: var(--serif);
      font-size: 1.1rem;
      font-style: italic;
      line-height: 1.7;
      color: rgba(216,208,192,0.95);
    }}

    .detail-feature {{
      display: grid;
      grid-template-columns: minmax(280px, 0.9fr) 1fr;
      gap: 1.5rem;
      align-items: end;
      margin: -1rem 0 3rem;
      cursor: zoom-in;
      border: 1px solid var(--border);
      background: linear-gradient(135deg, rgba(201,162,39,0.08), rgba(255,255,255,0.02));
      padding: 0.8rem;
    }}
    .detail-feature img {{
      width: 100%;
      aspect-ratio: 4/3;
      object-fit: cover;
      display: block;
    }}
    .detail-feature figcaption {{
      font-family: var(--serif);
      color: rgba(216,208,192,0.9);
      font-size: 1rem;
      line-height: 1.55;
      padding: 0.6rem 0.6rem 0.6rem 0;
    }}
    .detail-feature figcaption span {{
      display: block;
      font-family: var(--sans);
      font-size: 0.65rem;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--gold);
      margin-bottom: 0.6rem;
    }}
    @media (max-width: 800px) {{
      .detail-feature {{ grid-template-columns: 1fr; }}
      .detail-feature figcaption {{ padding: 0.4rem; }}
    }}

    .two-col {{
      display: grid;
      grid-template-columns: 1fr 340px;
      gap: 3rem;
      margin-bottom: 3rem;
    }}
    @media (max-width: 800px) {{
      .two-col {{ grid-template-columns: 1fr; }}
    }}

    .section-heading {{
      font-family: var(--sans);
      font-weight: 600;
      font-size: 0.65rem;
      letter-spacing: 0.2em;
      text-transform: uppercase;
      color: var(--gold);
      margin-bottom: 1rem;
      margin-top: 2rem;
      padding-bottom: 0.4rem;
      border-bottom: 1px solid var(--border);
    }}
    .section-heading:first-child {{ margin-top: 0; }}

    .col-text p {{
      font-family: var(--serif);
      font-size: 1.05rem;
      color: var(--text);
      margin-bottom: 1rem;
    }}
    .conservation-note {{
      font-size: 0.9rem !important;
      color: var(--text-dim) !important;
      font-style: italic;
    }}

    /* ── Facts ── */
    .col-facts {{ position: sticky; top: 2rem; align-self: start; }}
    .facts-list {{
      display: flex;
      flex-direction: column;
      gap: 0.6rem;
      margin-bottom: 2rem;
    }}
    .fact-item {{
      font-size: 0.88rem;
      color: var(--text);
      padding: 0.5rem 0;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: baseline;
      gap: 0.6rem;
    }}
    .fact-bullet {{ color: var(--gold); font-size: 0.5rem; flex-shrink: 0; }}

    /* ── Encounter Stats ── */
    .encounter-stats {{ margin-top: 0; }}
    .stat-row {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      padding: 0.5rem 0;
      border-bottom: 1px solid var(--border);
      gap: 1rem;
    }}
    .stat-label {{
      font-size: 0.75rem;
      letter-spacing: 0.05em;
      color: var(--text-dim);
    }}
    .stat-value {{
      font-size: 0.88rem;
      color: var(--text);
      text-align: right;
      color: var(--gold-light);
    }}

    /* ── Photo Gallery ── */
    .gallery-heading {{ margin-top: 3rem; }}
    .photo-gallery {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 4px;
      margin-top: 1rem;
    }}
    .gallery-item {{
      position: relative;
      aspect-ratio: 3/2;
      overflow: hidden;
      cursor: pointer;
      background: #111;
    }}
    .gallery-item img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      transition: transform 0.4s ease;
      display: block;
    }}
    .gallery-item:hover img {{ transform: scale(1.05); }}
    .gallery-caption {{
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
      padding: 0.6rem 0.8rem;
      background: linear-gradient(transparent, rgba(0,0,0,0.85));
      transform: translateY(100%);
      transition: transform 0.3s ease;
    }}
    .gallery-item:hover .gallery-caption {{ transform: translateY(0); }}
    .gallery-score {{ color: var(--gold); font-size: 0.75rem; }}
    .gallery-behaviour {{ font-size: 0.75rem; color: var(--text); margin-top: 0.2rem; }}

    /* ── Dividers ── */
    .section-divider {{
      height: 4px;
      background: linear-gradient(90deg, var(--gold) 0%, transparent 100%);
    }}

    /* ── Lightbox ── */
    .lightbox {{
      display: none;
      position: fixed;
      inset: 0;
      z-index: 1000;
      background: rgba(0,0,0,0.95);
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 2rem;
    }}
    .lightbox.active {{ display: flex; }}
    .lightbox-img {{
      max-width: 90vw;
      max-height: 80vh;
      object-fit: contain;
      border: 1px solid #222;
    }}
    .lightbox-caption {{
      margin-top: 1rem;
      text-align: center;
      max-width: 600px;
    }}
    .lightbox-behaviour {{
      font-family: var(--serif);
      font-style: italic;
      font-size: 1.1rem;
      color: var(--text);
    }}
    .lightbox-notes {{
      font-size: 0.85rem;
      color: var(--text-dim);
      margin-top: 0.4rem;
    }}
    .lightbox-close {{
      position: absolute;
      top: 1.5rem;
      right: 1.5rem;
      background: none;
      border: 1px solid #444;
      color: var(--text);
      font-size: 1.2rem;
      width: 42px;
      height: 42px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: 50%;
      transition: all 0.2s;
    }}
    .lightbox-close:hover {{ border-color: var(--gold); color: var(--gold); }}

    /* ── Footer ── */
    .site-footer {{
      background: #050505;
      border-top: 1px solid var(--border);
      padding: 3rem 2rem;
      text-align: center;
    }}
    .footer-rule {{ width: 60px; height: 2px; background: var(--gold); margin: 0 auto 1.5rem; }}
    .footer-title {{
      font-family: var(--serif);
      font-size: 1.3rem;
      color: var(--text-dim);
      margin-bottom: 0.5rem;
    }}
    .footer-sub {{
      font-size: 0.75rem;
      letter-spacing: 0.15em;
      text-transform: uppercase;
      color: var(--border);
    }}

    /* ── Big Five: cover badge ── */
    .cover-big5 {{
      margin: -0.5rem auto 2.5rem;
      padding: 0.45rem 1.4rem;
      border: 1px solid var(--gold);
      border-radius: 2px;
      color: var(--gold-light);
      font-size: 0.7rem;
      font-weight: 600;
      letter-spacing: 0.22em;
      text-transform: uppercase;
      background: rgba(201,162,39,0.06);
    }}

    /* ── Big Five: showcase band ── */
    .bigfive-band {{
      background: linear-gradient(180deg, #050505, var(--dark));
      padding: 4.5rem 2rem 5rem;
      border-top: 1px solid var(--border);
      border-bottom: 1px solid var(--border);
    }}
    .bigfive-head {{ text-align: center; max-width: 720px; margin: 0 auto 2.5rem; }}
    .bigfive-eyebrow {{
      font-size: 0.72rem;
      font-weight: 600;
      letter-spacing: 0.25em;
      text-transform: uppercase;
      color: var(--gold);
      margin-bottom: 1rem;
    }}
    .bigfive-title {{
      font-family: var(--serif);
      font-size: clamp(1.8rem, 4vw, 3rem);
      font-weight: 700;
      color: #fff;
      margin-bottom: 0.8rem;
    }}
    .bigfive-sub {{
      font-family: var(--serif);
      font-style: italic;
      color: var(--text-dim);
      font-size: 1.05rem;
    }}
    .bigfive-grid {{
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 6px;
      max-width: 1400px;
      margin: 0 auto;
    }}
    @media (max-width: 900px) {{
      .bigfive-grid {{ grid-template-columns: repeat(2, 1fr); }}
    }}
    @media (max-width: 520px) {{
      .bigfive-grid {{ grid-template-columns: 1fr; }}
    }}
    .bf-tile {{
      position: relative;
      display: block;
      aspect-ratio: 3/4;
      background-size: cover;
      background-position: center;
      overflow: hidden;
      text-decoration: none;
    }}
    .bf-tile-overlay {{
      position: absolute;
      inset: 0;
      background: linear-gradient(transparent 35%, rgba(0,0,0,0.85));
      transition: background 0.3s ease;
    }}
    .bf-tile:hover .bf-tile-overlay {{ background: linear-gradient(rgba(201,162,39,0.12), rgba(0,0,0,0.85)); }}
    .bf-tile-name {{
      position: absolute;
      bottom: 1rem;
      left: 0;
      right: 0;
      text-align: center;
      font-family: var(--serif);
      font-size: 1.1rem;
      color: #fff;
      letter-spacing: 0.02em;
      text-shadow: 0 1px 8px rgba(0,0,0,0.8);
    }}

    /* ── Big Five: hero ribbon + nav star ── */
    .hero-big5 {{
      display: inline-block;
      margin-bottom: 0.8rem;
      padding: 0.3rem 0.9rem;
      background: var(--gold);
      color: #1a1200;
      font-family: var(--sans);
      font-size: 0.62rem;
      font-weight: 600;
      letter-spacing: 0.2em;
      text-transform: uppercase;
      border-radius: 2px;
    }}
    .nav-big5 {{ color: var(--gold) !important; }}

    /* Scrollbar */
    ::-webkit-scrollbar {{ width: 6px; }}
    ::-webkit-scrollbar-track {{ background: var(--black); }}
    ::-webkit-scrollbar-thumb {{ background: #333; border-radius: 3px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: var(--gold); }}
  </style>
</head>
<body>

<!-- Side Navigation -->
<nav class="side-nav" aria-label="Animal navigation">
  <ul>
    {nav_items}
  </ul>
</nav>

<!-- Cover -->
<header class="cover">
  <div class="cover-rule"></div>
  <div class="cover-eyebrow">A Wildlife Field Journal</div>
  <h1 class="cover-title">Safari</h1>
  <p class="cover-subtitle">An intimate encounter with Africa's wild creatures, told through the lens</p>
  {'<div class="cover-big5">★ Big Five — Complete Set ★</div>' if len(big_five_present) == 5 else ''}
  <div class="cover-stats">
    <div class="cover-stat">
      <div class="cover-stat-num">{all_animals_count}</div>
      <div class="cover-stat-label">Species</div>
    </div>
    <div class="cover-stat">
      <div class="cover-stat-num">{total_photos_analysed}</div>
      <div class="cover-stat-label">Best Shots</div>
    </div>
    <div class="cover-stat">
      <div class="cover-stat-num">{all_source_photo_count}</div>
      <div class="cover-stat-label">Photos Analysed</div>
    </div>
  </div>
  <div class="scroll-hint">↓ &nbsp; scroll to explore</div>
</header>
{big_five_band}

<!-- Animal Sections -->
{"".join(f'<div class="section-divider"></div>{make_animal_section(a, selected[a], animal_info)}' for a in animals)}

<!-- Footer -->
<footer class="site-footer">
  <div class="footer-rule"></div>
  <p class="footer-title">Safari Field Journal</p>
  <p class="footer-sub">Created with Claude · {all_animals_count} species · {total_photos_analysed} photographs</p>
</footer>

<!-- Lightbox -->
<div class="lightbox" id="lightbox" onclick="closeLightbox(event)">
  <button class="lightbox-close" onclick="closeLightbox()">&times;</button>
  <img class="lightbox-img" id="lightbox-img" src="" alt="">
  <div class="lightbox-caption">
    <p class="lightbox-behaviour" id="lightbox-behaviour"></p>
    <p class="lightbox-notes" id="lightbox-notes"></p>
  </div>
</div>

<script>
  function openLightbox(src, detailSrc, behaviour, notes) {{
    document.getElementById('lightbox-img').src = detailSrc || src;
    document.getElementById('lightbox-behaviour').textContent = behaviour;
    document.getElementById('lightbox-notes').textContent = notes;
    document.getElementById('lightbox').classList.add('active');
    document.body.style.overflow = 'hidden';
  }}

  function closeLightbox(e) {{
    if (e && e.target !== document.getElementById('lightbox') &&
        !e.target.classList.contains('lightbox-close')) return;
    document.getElementById('lightbox').classList.remove('active');
    document.body.style.overflow = '';
  }}

  document.addEventListener('keydown', e => {{
    if (e.key === 'Escape') {{
      document.getElementById('lightbox').classList.remove('active');
      document.body.style.overflow = '';
    }}
  }});

  // Highlight nav on scroll
  const sections = document.querySelectorAll('.animal-section');
  const navLinks = document.querySelectorAll('.nav-link');
  const observer = new IntersectionObserver(entries => {{
    entries.forEach(entry => {{
      if (entry.isIntersecting) {{
        const id = entry.target.id;
        navLinks.forEach(l => {{
          l.style.color = '';
          l.style.borderLeftColor = '';
        }});
        const active = document.querySelector(`.nav-link[href="#${{id}}"]`);
        if (active) {{
          active.style.color = 'var(--gold-light)';
          active.style.borderLeftColor = 'var(--gold-light)';
        }}
      }}
    }});
  }}, {{ threshold: 0.3 }});
  sections.forEach(s => observer.observe(s));
</script>
</body>
</html>"""
    return html


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    print("=" * 60)
    print("  SAFARI PHOTO LOGBOOK PIPELINE")
    print("=" * 60)

    # Stage 1: Classify all photos
    classifications = await run_classifications()

    # Stage 2: Select best per animal
    selected = select_best_photos(classifications)

    # Stage 3: Generate web assets
    generate_web_assets(selected)

    # Stage 4: Animal background info
    animal_info = generate_animal_info(selected)

    # Stage 5: Build HTML
    html = build_html(selected, animal_info)
    out_path = Path.home() / "safari" / "index.html"
    out_path.write_text(html, encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"  DONE! Open your logbook:")
    print(f"  file://{out_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
