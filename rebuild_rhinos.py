#!/usr/bin/env python3
"""Promote this morning's best White Rhino frames into the logbook and rebuild.

The morning rhinos (DSC_6968–6995, in new-raw-photo2) were correctly classified
but tied at q8 with older frames and lost the top-5 tie-break. Nudge the best of
them just above the old set so they lead the rhino section, then rebuild the site.
Runs fully offline: every animal is already cached in animal_info.json.
"""
from pathlib import Path

from pipeline import (
    load_json, save_json, CLASSIFICATIONS_FILE,
    select_best_photos, generate_web_assets, generate_animal_info, build_html,
)

# Visual ranking of the morning rhinos → distinct scores above the old q8 frames,
# so order is deterministic and DSC_6991 (dramatic head-on) takes the hero slot.
RHINO_PROMOTIONS = {
    "DSC_6991.JPG": 9.0,   # head-on, walking toward camera — hero
    "DSC_6990.JPG": 8.9,   # head-on, grazing, ears framed
    "DSC_6995.JPG": 8.8,   # vertical, golden light, full body
    "DSC_6979.JPG": 8.7,   # strong side profile, good horn
    "DSC_6973.JPG": 8.6,   # sharp side, fine detail
}


def main() -> None:
    classifications = load_json(CLASSIFICATIONS_FILE, {})
    for name, score in RHINO_PROMOTIONS.items():
        if name in classifications:
            classifications[name]["quality_score"] = score
            print(f"  promoted {name} → q{score}")
        else:
            print(f"  ! {name} not in classifications")
    save_json(CLASSIFICATIONS_FILE, classifications)

    selected = select_best_photos(classifications)
    generate_web_assets(selected)
    animal_info = generate_animal_info(selected)  # all cached → no API calls
    html = build_html(selected, animal_info)
    out_path = Path.home() / "safari" / "index.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"\nRebuilt → {out_path}")


if __name__ == "__main__":
    main()
