"""
    dataloader.py

                Raw DICOM

                ↓

                load_dicom()

                ↓

                NumPy

                ↓

                preprocess_image()

                ↓

                Processed NumPy

                ↓

                RSNADataset()

                ↓

                Tensor

                ↓

                DataLoader

                ↓

                Batch

"""

from torch.utils.data import DataLoader
from src.data.dataset import RSNADataset

def build_dataloader(
        patient_ids,
        labels,
        image_dir,
        batch_size = 16,
        shuffle = True,
        num_workers = 0,
):
    dataset = RSNADataset(
        patient_ids=patient_ids,
        labels=labels,
        image_dir=image_dir,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    return dataloader