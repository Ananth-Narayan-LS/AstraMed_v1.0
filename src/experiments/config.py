"""
Experiment configuration for AstraMed.

This module provides a centralized configuration object
for reproducible model experiments.
"""

from dataclasses import dataclass
from pathlib import Path

from src.utils.config import OUTPUT_DIR

import torch


def get_device() -> torch.device:
    """
    Select the best available computation device.

    Returns
    -------
    torch.device
        CUDA device when available, otherwise CPU.
    """

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

@dataclass
class ExperimentConfig:
    """
    Configuration for a single AstraMed experiment.

    Parameters
    ----------
    model_name : str
        Name of the model architecture.

    pretrained : bool
        Whether to use pretrained model weights.

    image_size : int
        Input image size.

    batch_size : int
        Number of samples processed in one batch.

    learning_rate : float
        Optimizer learning rate.

    epochs : int
        Number of training epochs.

    seed : int
        Random seed used for reproducibility.

    experiment_name : str or None
        Optional custom experiment name.

    output_dir : Path
        Root directory for experiment outputs.
    """


    model_name: str = "baseline_cnn"
    pretrained: bool = False

    # Dataset / input configuration
    dataset_image_size: int = 512
    model_image_size: int = 224

    use_imagenet_normalization: bool = True

    batch_size: int = 8

    learning_rate: float = 0.001
    epochs: int = 5

    seed: int = 42

    experiment_name: str | None = None

    output_dir: Path = OUTPUT_DIR

    def __post_init__(self):
        """
        Validate configuration and create an experiment name.
        """

        if self.dataset_image_size <= 0:
            raise ValueError(
                "dataset_image_size must be greater than 0."
            )

        if self.model_image_size <= 0:
            raise ValueError(
                "model_image_size must be greater than 0."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0."
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be greater than 0."
            )

        if self.epochs <= 0:
            raise ValueError(
                "epochs must be greater than 0."
            )

        if self.seed < 0:
            raise ValueError(
                "seed must be non-negative."
            )

        if self.experiment_name is None:
            self.experiment_name = (
                self._generate_experiment_name()
            )

        self.output_dir = Path(self.output_dir)


    def _generate_experiment_name(self) -> str:
        """
        Generate a deterministic experiment name.
        """

        pretrained_tag = (
            "pretrained"
            if self.pretrained
            else "scratch"
        )

        return (
            f"{self.model_name}"
            f"_{pretrained_tag}"
            f"_img{self.model_image_size}"
            f"_bs{self.batch_size}"
            f"_lr{self.learning_rate}"
            f"_ep{self.epochs}"
            f"_seed{self.seed}"
        )

    @property
    def experiment_dir(self) -> Path:
        """
        Directory dedicated to this experiment.
        """

        return self.output_dir / "experiments" / self.experiment_name

    @property
    def checkpoint_dir(self) -> Path:
        """
        Directory for experiment checkpoints.
        """

        return self.experiment_dir / "checkpoints"

    @property
    def log_dir(self) -> Path:
        """
        Directory for experiment logs.
        """

        return self.experiment_dir / "logs"

    def create_directories(self) -> None:
        """
        Create all directories required by the experiment.
        """

        self.experiment_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.log_dir.mkdir(
            parents=True,
            exist_ok=True,
        )