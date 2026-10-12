"""
Multimodal dataset: image + structured clinical features (+ text fields, always empty for RSNA).

Sample schema (a dict):
    patient_id, image [1,224,224], target (int), age (float or NaN), sex (str or ""),
    symptoms_text (""), clinical_history_text (""), clinical_features [struct_dim],
    modality_mask [4] = (image, structured, symptoms, history), bounding_boxes (list).
"""
from __future__ import annotations

import math

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.multimodal.clinical_data import ClinicalFeatureEncoder, validate_join


class MultimodalDataset(Dataset):
    def __init__(
        self,
        patient_ids,
        labels: dict,
        clinical_table: pd.DataFrame,
        encoder: ClinicalFeatureEncoder,
        image_store,
        forbidden_ids=None,
    ):
        patient_ids = list(patient_ids)
        if len(set(patient_ids)) != len(patient_ids):
            raise ValueError("Duplicate patient IDs in the requested split.")
        if forbidden_ids is not None and set(patient_ids) & set(forbidden_ids):
            raise ValueError("Requested split overlaps forbidden (held-out) patients.")
        validate_join(clinical_table, labels.keys())

        table = clinical_table.set_index("patient_id").loc[patient_ids].reset_index()
        self.patient_ids = patient_ids
        self.labels = labels
        self.image_store = image_store
        self.features = torch.from_numpy(encoder.transform(table))
        self.structured_mask = torch.from_numpy(encoder.structured_mask(table))
        self.ages = table["age"].tolist()
        self.sexes = [("" if s is None or (isinstance(s, float) and math.isnan(s)) else s)
                      for s in table["sex"].tolist()]

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, index: int) -> dict:
        patient_id = self.patient_ids[index]
        label = self.labels[patient_id]
        mask = torch.tensor([1.0, float(self.structured_mask[index]), 0.0, 0.0])
        return {
            "patient_id": patient_id,
            "image": self.image_store.get(patient_id),
            "target": int(label["target"]),
            "age": float(self.ages[index]) if not pd.isna(self.ages[index]) else float("nan"),
            "sex": self.sexes[index],
            "symptoms_text": "",
            "clinical_history_text": "",
            "clinical_features": self.features[index],
            "modality_mask": mask,
            "bounding_boxes": label["bounding_boxes"],
        }


def collate_batch(samples: list[dict]) -> dict:
    """Stack tensors; keep ids/text as lists. Bounding boxes are dropped from training batches."""
    return {
        "patient_id": [s["patient_id"] for s in samples],
        "image": torch.stack([s["image"] for s in samples]),
        "target": torch.tensor([s["target"] for s in samples], dtype=torch.float32),
        "clinical_features": torch.stack([s["clinical_features"] for s in samples]),
        "modality_mask": torch.stack([s["modality_mask"] for s in samples]),
        "symptoms_text": [s["symptoms_text"] for s in samples],
        "clinical_history_text": [s["clinical_history_text"] for s in samples],
    }


def move_to_device(batch: dict, device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
