"""
checkpoint.py

Utilities for saving and loading AstraMed model checkpoints.
"""

from pathlib import Path

import torch


def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    path,
):
    """
    Save model and optimizer state.

    Parameters
    ----------
    model : torch.nn.Module
        Model to save.

    optimizer : torch.optim.Optimizer
        Optimizer associated with the model.

    epoch : int
        Epoch number.

    loss : float
        Validation or training loss at the checkpoint.

    path : str or Path
        Location where checkpoint will be saved.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    torch.save(checkpoint, path)

    print(f"Checkpoint saved to: {path}")


def load_checkpoint(
    model,
    optimizer,
    path,
    device,
):
    """
    Load model and optimizer state from checkpoint.

    Returns
    -------
    dict
        Checkpoint information.
    """

    path = Path(path)

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    print(f"Checkpoint loaded from: {path}")

    return checkpoint