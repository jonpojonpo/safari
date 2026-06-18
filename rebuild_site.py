#!/usr/bin/env python3
"""Full offline site rebuild: regenerate every web asset with the centred crop
and rebuild the Big Five-led HTML. All animal info is already cached, so no API."""
from pathlib import Path

from pipeline import (
    load_json, save_json, CLASSIFICATIONS_FILE,
    HEROES_DIR, DETAILS_DIR, THUMBS_DIR,
    select_best_photos, generate_web_assets, generate_animal_info, build_html,
    ANIMAL_INFO_FILE,
)


def main() -> None:
    c = load_json(CLASSIFICATIONS_FILE, {})

    # Merge the stray "Rhino" label into the Big Five "White Rhino".
    merged = 0
    for d in c.values():
        if d.get("animal") == "Rhino":
            d["animal"] = "White Rhino"
            d.setdefault("scientific_name", "Ceratotherium simum")
            merged += 1
    if merged:
        save_json(CLASSIFICATIONS_FILE, c)
        print(f"Merged {merged} 'Rhino' → 'White Rhino'")

    # Wipe generated assets so they are all recut with the centred crop.
    removed = 0
    for d in (HEROES_DIR, DETAILS_DIR, THUMBS_DIR):
        for f in d.glob("*.jpg"):
            f.unlink()
            removed += 1
    print(f"Removed {removed} old asset files")

    selected = select_best_photos(c)
    generate_web_assets(selected)
    animal_info = generate_animal_info(selected)  # cached → offline
    html = build_html(selected, animal_info)
    Path(__file__).resolve().parent.joinpath("index.html").write_text(html, encoding="utf-8")
    print("\nRebuilt index.html")


if __name__ == "__main__":
    main()
