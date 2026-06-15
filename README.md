# Safari photo logbook

The existing `pipeline.py` builds the web journal using Claude-generated labels.
`local_pipeline.py` adds a fully local classification, rating, and selection pass.

## Local classifier

It combines:

- CLIP image embeddings running on Apple Metal (MPS)
- k-nearest-neighbour classification from the existing 447 labelled photos
- pixel-level sharpness, exposure, contrast, and colour measurements
- diversity-aware selection that suppresses near-identical burst frames

The original photos are only read. Downsampling happens in memory.

```bash
uv sync
uv run python local_pipeline.py
```

The first run downloads `openai/clip-vit-base-patch32`. Later runs can prohibit
network access:

```bash
uv run python local_pipeline.py --offline
```

Outputs:

- `data/local_embeddings.npz`: reusable image embeddings
- `data/local_classifications.json`: local animal and quality predictions
- `data/local_selection.json`: top five diverse photos per animal

To preserve the existing animal labels while using the better burst-aware
selection:

```bash
uv run python local_pipeline.py --selection-only --select-from existing
```
