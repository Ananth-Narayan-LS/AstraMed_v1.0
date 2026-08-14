"""
dataset.py

PyTorch Dataset for RSNA Pneumonia Detection.

"""

from pathlib import Path
import torch
from torch.utils.data import Dataset

from src.data.dicom_loader import load_dicom
from src.data.preprocessing import preprocess_image

import torch


class RSNADataset(Dataset):
    def __init__(
            self,
            patient_ids,
            labels,
            image_dir,
    ):
        self.patient_ids = patient_ids
        self.labels = labels
        self.image_dir = Path(image_dir)

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]

        image_path = self.image_dir / f"{patient_id}.dcm"

        image,_ = load_dicom(image_path)

        image = preprocess_image(image)

        image = torch.from_numpy(image).float()

        image = image.unsqueeze(0)

        sample = {
            "image" : image,
            "target" : self.labels[patient_id]["target"],
            "patient_ids" : patient_id
        }

        return sample