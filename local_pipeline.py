#!/usr/bin/env python3
"""Local wildlife classification, quality scoring, and diverse selection.

The pretrained image encoder is downloaded once and then runs entirely locally.
Existing classifications act as the labelled reference set; no cloud API is used.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoModel, AutoProcessor


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_PHOTO_DIRS = (
    Path.home() / "Movies",
    DATA_DIR / "new-raw-photos",
    DATA_DIR / "new-raw-photo2",
    DATA_DIR / "new-raw-photo3",
)
DEFAULT_MODEL = "openai/clip-vit-base-patch32"
CLASSIFICATIONS_FILE = DATA_DIR / "classifications.json"
LOCAL_CLASSIFICATIONS_FILE = DATA_DIR / "local_classifications.json"
EMBEDDING_CACHE_FILE = DATA_DIR / "local_embeddings.npz"
SELECTION_FILE = DATA_DIR / "local_selection.json"

MIN_REFERENCE_EXAMPLES = 2
K_NEIGHBOURS = 9

# Collapse inconsistent cloud labels before they become local training classes.
CANONICAL_LABELS = {
    "rhino": "White Rhino",
    "kingfisher": "Malachite Kingfisher",
    "bulbul": "Dark-capped Bulbul",
    "vulture": "Vulture",
    "white-backed vulture": "White-backed Vulture",
    "turkey vulture": "Vulture",
    "tufted titmouse": "Bird",
    "chital": "Impala",
    "tiger": "Unknown",
}

EXCLUDED_LABELS = {"Unknown", "No Animal", "None", "Bird"}


@dataclass(frozen=True)
class PhotoMetrics:
    sharpness: float
    exposure: float
    contrast: float
    color: float
    technical_score: float


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    temporary.replace(path)


def canonical_label(label: str) -> str:
    cleaned = label.strip()
    return CANONICAL_LABELS.get(cleaned.casefold(), cleaned)


def discover_photos(photo_dirs: list[Path]) -> dict[str, Path]:
    photos: dict[str, Path] = {}
    for directory in photo_dirs:
        if not directory.exists():
            continue
        for pattern in ("*.JPG", "*.JPEG", "*.jpg", "*.jpeg", "*.PNG", "*.png"):
            for path in sorted(directory.glob(pattern)):
                photos.setdefault(path.name, path)
    return photos


def choose_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_encoder(model_name: str, local_only: bool = False):
    device = choose_device()
    print(f"Loading {model_name} on {device}...")
    processor = AutoProcessor.from_pretrained(model_name, local_files_only=local_only)
    model = AutoModel.from_pretrained(model_name, local_files_only=local_only)
    model.eval().to(device)
    return processor, model, device


def load_embedding_cache(path: Path, model_name: str) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    if not path.exists():
        return {}, {}
    try:
        cache = np.load(path, allow_pickle=False)
        cached_model = str(cache["model_name"].item())
        if cached_model != model_name:
            return {}, {}
        names = cache["names"].tolist()
        fingerprints = cache["fingerprints"].tolist()
        vectors = cache["vectors"].astype(np.float32)
        return dict(zip(names, vectors, strict=True)), dict(zip(names, fingerprints, strict=True))
    except (OSError, ValueError, KeyError):
        return {}, {}


def save_embedding_cache(
    path: Path,
    model_name: str,
    vectors: dict[str, np.ndarray],
    fingerprints: dict[str, str],
) -> None:
    names = sorted(vectors)
    matrix = np.stack([vectors[name] for name in names]).astype(np.float16)
    np.savez_compressed(
        path,
        model_name=np.array(model_name),
        names=np.array(names),
        fingerprints=np.array([fingerprints[name] for name in names]),
        vectors=matrix,
    )


def fingerprint(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def embed_photos(
    photos: dict[str, Path],
    processor: Any,
    model: Any,
    device: str,
    cache_path: Path,
    model_name: str,
    batch_size: int,
) -> dict[str, np.ndarray]:
    vectors, fingerprints = load_embedding_cache(cache_path, model_name)
    todo = [
        (name, path)
        for name, path in photos.items()
        if name not in vectors or fingerprints.get(name) != fingerprint(path)
    ]
    if not todo:
        print(f"Embeddings: all {len(photos)} loaded from cache.")
        return {name: vectors[name] for name in photos}

    print(f"Embedding {len(todo)} photos ({len(vectors)} cached)...")
    with torch.inference_mode():
        for start in range(0, len(todo), batch_size):
            batch = todo[start : start + batch_size]
            images = []
            for _, path in batch:
                with Image.open(path) as image:
                    images.append(ImageOps.exif_transpose(image).convert("RGB"))
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            features = model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            features_np = features.float().cpu().numpy()
            for (name, path), vector in zip(batch, features_np, strict=True):
                vectors[name] = vector
                fingerprints[name] = fingerprint(path)
            print(f"  {min(start + len(batch), len(todo))}/{len(todo)}")

    live_names = set(photos)
    vectors = {name: vector for name, vector in vectors.items() if name in live_names}
    fingerprints = {name: value for name, value in fingerprints.items() if name in live_names}
    save_embedding_cache(cache_path, model_name, vectors, fingerprints)
    return vectors


def build_reference_index(
    classifications: dict[str, dict[str, Any]],
    vectors: dict[str, np.ndarray],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    counts = Counter(
        canonical_label(item.get("animal", "Unknown"))
        for name, item in classifications.items()
        if name in vectors
    )
    names = []
    labels = []
    quality = []
    for name, item in classifications.items():
        label = canonical_label(item.get("animal", "Unknown"))
        if name not in vectors or label in EXCLUDED_LABELS or counts[label] < MIN_REFERENCE_EXAMPLES:
            continue
        names.append(name)
        labels.append(label)
        quality.append(float(item.get("quality_score", 5)))
    if not names:
        raise RuntimeError("No usable labelled reference photos were found.")
    matrix = np.stack([vectors[name] for name in names])
    return labels, matrix, np.asarray(quality, dtype=np.float32), np.asarray(names)


def classify_from_neighbours(
    vector: np.ndarray,
    labels: list[str],
    reference_vectors: np.ndarray,
    reference_quality: np.ndarray,
    reference_names: np.ndarray,
    exclude_name: str | None = None,
) -> tuple[str, float, float, list[dict[str, Any]]]:
    similarities = reference_vectors @ vector
    if exclude_name is not None:
        similarities = similarities.copy()
        similarities[reference_names == exclude_name] = -1.0
    k = min(K_NEIGHBOURS, len(similarities))
    nearest = np.argpartition(similarities, -k)[-k:]
    nearest = nearest[np.argsort(similarities[nearest])[::-1]]

    votes: defaultdict[str, float] = defaultdict(float)
    for index in nearest:
        # CLIP similarities are compressed; exponential weighting rewards close matches.
        votes[labels[index]] += math.exp(float(similarities[index]) * 12.0)
    ranked_votes = sorted(votes.items(), key=lambda item: item[1], reverse=True)
    winner, winner_vote = ranked_votes[0]
    confidence = winner_vote / sum(votes.values())

    same_class = [index for index in nearest if labels[index] == winner]
    quality_weights = np.exp(np.asarray([similarities[index] for index in same_class]) * 12.0)
    predicted_quality = float(
        np.average(reference_quality[same_class], weights=quality_weights)
    )
    neighbours = [
        {
            "file": str(reference_names[index]),
            "animal": labels[index],
            "similarity": round(float(similarities[index]), 4),
        }
        for index in nearest[:5]
    ]
    return winner, confidence, predicted_quality, neighbours


def image_metrics(path: Path) -> PhotoMetrics:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        rgb = np.asarray(image, dtype=np.float32) / 255.0

    luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    # Variance of a discrete Laplacian is a useful resolution-independent blur signal.
    laplacian = (
        -4.0 * luminance[1:-1, 1:-1]
        + luminance[:-2, 1:-1]
        + luminance[2:, 1:-1]
        + luminance[1:-1, :-2]
        + luminance[1:-1, 2:]
    )
    lap_variance = float(np.var(laplacian))
    sharpness = float(np.clip((math.log10(lap_variance + 1e-7) + 5.0) / 2.5, 0, 1))

    mean_luma = float(np.mean(luminance))
    clipped = float(np.mean((luminance < 0.02) | (luminance > 0.98)))
    exposure = float(np.clip(1.0 - abs(mean_luma - 0.48) / 0.48 - clipped * 2.5, 0, 1))

    p5, p95 = np.percentile(luminance, [5, 95])
    contrast = float(np.clip((p95 - p5) / 0.65, 0, 1))
    saturation = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    mean_saturation = float(np.mean(saturation))
    color = float(np.clip(1.0 - abs(mean_saturation - 0.28) / 0.4, 0, 1))

    weighted = 0.45 * sharpness + 0.3 * exposure + 0.15 * contrast + 0.1 * color
    technical_score = 1.0 + 9.0 * weighted
    return PhotoMetrics(sharpness, exposure, contrast, color, technical_score)


def rate_photo(metrics: PhotoMetrics, neighbour_quality: float) -> float:
    # Neighbours supply aesthetic/subject-interest signal; pixels supply technical truth.
    score = 0.58 * neighbour_quality + 0.42 * metrics.technical_score
    return round(float(np.clip(score, 1, 10)), 1)


def analyse(
    photos: dict[str, Path],
    classifications: dict[str, dict[str, Any]],
    vectors: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    labels, reference_vectors, reference_quality, reference_names = build_reference_index(
        classifications, vectors
    )
    output: dict[str, dict[str, Any]] = {}
    print(f"Classifying and rating {len(photos)} photos locally...")
    for index, (name, path) in enumerate(photos.items(), 1):
        animal, confidence, neighbour_quality, neighbours = classify_from_neighbours(
            vectors[name],
            labels,
            reference_vectors,
            reference_quality,
            reference_names,
            exclude_name=name,
        )
        metrics = image_metrics(path)
        output[name] = {
            "file": name,
            "animal": animal,
            "confidence": round(confidence, 4),
            "quality_score": rate_photo(metrics, neighbour_quality),
            "technical_score": round(metrics.technical_score, 1),
            "metrics": {
                "sharpness": round(metrics.sharpness, 4),
                "exposure": round(metrics.exposure, 4),
                "contrast": round(metrics.contrast, 4),
                "color": round(metrics.color, 4),
            },
            "nearest_references": neighbours,
            "source": "local-clip-knn",
        }
        if index % 25 == 0 or index == len(photos):
            print(f"  {index}/{len(photos)}")
    return output


def select_diverse(
    classifications: dict[str, dict[str, Any]],
    vectors: dict[str, np.ndarray],
    top_n: int,
    similarity_penalty: float,
) -> dict[str, list[dict[str, Any]]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for name, item in classifications.items():
        animal = canonical_label(item.get("animal", "Unknown"))
        if name in vectors and animal not in EXCLUDED_LABELS:
            grouped[animal].append(item)

    selections: dict[str, list[dict[str, Any]]] = {}
    for animal, candidates in sorted(grouped.items()):
        remaining = sorted(candidates, key=lambda item: item.get("quality_score", 0), reverse=True)
        chosen: list[dict[str, Any]] = []
        while remaining and len(chosen) < top_n:
            if not chosen:
                best = remaining[0]
            else:
                chosen_vectors = np.stack([vectors[item["file"]] for item in chosen])

                def utility(item: dict[str, Any]) -> float:
                    duplicate_similarity = float(np.max(chosen_vectors @ vectors[item["file"]]))
                    return float(item.get("quality_score", 0)) - similarity_penalty * max(
                        0.0, duplicate_similarity - 0.82
                    ) / 0.18

                best = max(remaining, key=utility)
            chosen.append(best)
            remaining.remove(best)
        selections[animal] = chosen
    return selections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--photos",
        action="append",
        type=Path,
        help="Photo directory; may be supplied more than once (defaults to Movies and data/new-raw-photos).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--top", type=int, default=5, help="Photos selected per animal.")
    parser.add_argument(
        "--select-from",
        choices=("local", "existing"),
        default="local",
        help="Use local predictions or existing labels for the diverse shortlist.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require the model to already exist in the local Hugging Face cache.",
    )
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Skip classification and rebuild only the diverse shortlist.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    photo_dirs = args.photos or list(DEFAULT_PHOTO_DIRS)
    photos = discover_photos(photo_dirs)
    if not photos:
        print("No photos found.", file=sys.stderr)
        return 1
    existing = load_json(CLASSIFICATIONS_FILE, {})
    if not existing:
        print(f"Missing reference labels: {CLASSIFICATIONS_FILE}", file=sys.stderr)
        return 1

    processor, model, device = load_encoder(args.model, args.offline)
    vectors = embed_photos(
        photos,
        processor,
        model,
        device,
        EMBEDDING_CACHE_FILE,
        args.model,
        args.batch_size,
    )
    del model
    if device == "mps":
        torch.mps.empty_cache()

    local = load_json(LOCAL_CLASSIFICATIONS_FILE, {})
    if not args.selection_only:
        local = analyse(photos, existing, vectors)
        save_json(LOCAL_CLASSIFICATIONS_FILE, local)
        print(f"Saved local analysis to {LOCAL_CLASSIFICATIONS_FILE}")

    selection_source = local if args.select_from == "local" else existing
    if not selection_source:
        print("No classifications available for selection.", file=sys.stderr)
        return 1
    selected = select_diverse(selection_source, vectors, args.top, similarity_penalty=3.0)
    save_json(SELECTION_FILE, selected)
    print(f"Selected {sum(map(len, selected.values()))} photos across {len(selected)} animals.")
    print(f"Saved diverse shortlist to {SELECTION_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
