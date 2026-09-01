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


# dataloader.py takes your custom RSNADataset and converts it into batches of images and labels that the CNN can efficiently train on.

from torch.utils.data import DataLoader
from src.data.dataset import RSNADataset


def build_dataloader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )