"""
Build the local train/validation image cache for Sprint 8 (test patients are NOT cached).

    python scripts/build_image_cache.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.labels import load_labels
from src.data.split_data import split_patients
from src.multimodal.image_cache import build_cache
from src.utils.config import OUTPUT_DIR, TRAIN_IMAGES, TRAIN_LABELS

CACHE_DIR = OUTPUT_DIR / "multimodal" / "cache"


def main():
    labels, _ = load_labels(TRAIN_LABELS)
    train_ids, val_ids, _test_ids_never_used = split_patients(labels)
    for name, ids in (("val", val_ids), ("train", train_ids)):
        print(f"Caching {name}: {len(ids)} patients", flush=True)
        build_cache(ids, TRAIN_IMAGES, CACHE_DIR / f"images_{name}_224_fp16")
    print("Done:", CACHE_DIR)


if __name__ == "__main__":
    main()
