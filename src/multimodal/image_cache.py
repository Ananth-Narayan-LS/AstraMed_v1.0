"""
Local cache of preprocessed chest X-rays so multimodal experiments do not decode
1024x1024 DICOMs on every epoch.

Each cached image is exactly what the existing pipeline feeds to the image encoder
just before channel replication and ImageNet normalization:
    DICOM -> preprocess_image (min-max, 512x512) -> torchvision Resize((224, 224))
stored as float16 in [0, 1]. The existing AdvancedInputAdapter(224) then replicates
channels and normalizes (its resize step is a no-op at 224x224).

Only TRAIN and VALIDATION patients are cached. Test patients are never read here.
"""
from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

from src.data.dicom_loader import load_dicom
from src.data.preprocessing import preprocess_image

CACHE_IMAGE_SIZE = 224


def preprocess_for_cache(path) -> np.ndarray:
    image, _ = load_dicom(path)
    processed = preprocess_image(image)
    tensor = torch.from_numpy(processed).float().unsqueeze(0).unsqueeze(0)
    tensor = transforms.Resize((CACHE_IMAGE_SIZE, CACHE_IMAGE_SIZE))(tensor)
    return tensor[0, 0].numpy().astype(np.float16)


def _worker(path):
    return preprocess_for_cache(path)


def build_cache(patient_ids, image_dir, out_prefix: Path, workers: int = 10) -> None:
    out_prefix = Path(out_prefix)
    array_path = out_prefix.with_suffix(".npy")
    ids_path = out_prefix.with_suffix(".ids.json")
    for path in (array_path, ids_path):
        if path.exists():
            raise FileExistsError(f"{path} already exists; refusing to overwrite it.")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    patient_ids = list(patient_ids)
    paths = [Path(image_dir) / f"{pid}.dcm" for pid in patient_ids]
    array = np.lib.format.open_memmap(
        array_path, mode="w+", dtype=np.float16,
        shape=(len(paths), CACHE_IMAGE_SIZE, CACHE_IMAGE_SIZE),
    )
    with Pool(workers) as pool:
        for index, image in enumerate(pool.imap(_worker, paths, chunksize=16)):
            array[index] = image
            if (index + 1) % 2000 == 0:
                print(f"  cached {index + 1}/{len(paths)}", flush=True)
    array.flush()
    ids_path.write_text(json.dumps(patient_ids), encoding="utf-8")


class CachedImageStore:
    """Read-only access to a cache built by build_cache()."""

    def __init__(self, out_prefix):
        out_prefix = Path(out_prefix)
        self.array = np.load(out_prefix.with_suffix(".npy"), mmap_mode="r")
        self.ids = json.loads(out_prefix.with_suffix(".ids.json").read_text(encoding="utf-8"))
        if len(self.ids) != self.array.shape[0]:
            raise ValueError("Cache is corrupt: id list and image array differ in length.")
        self.index = {pid: i for i, pid in enumerate(self.ids)}

    def get(self, patient_id: str) -> torch.Tensor:
        """Return a [1, 224, 224] float32 tensor."""
        row = self.index[patient_id]
        return torch.from_numpy(np.asarray(self.array[row], dtype=np.float32)).unsqueeze(0)
