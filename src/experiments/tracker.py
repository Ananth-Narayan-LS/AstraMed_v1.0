"""
Experiment tracking utilities for AstraMed.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class ExperimentTracker:
    """
    Track configuration, model metadata, metrics,
    and experiment status.
    """

    def __init__(self, experiment_dir: str | Path):
        self.experiment_dir = Path(experiment_dir)

        self.experiment_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.history: list[dict[str, Any]] = []

        self.metadata: dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "status": "initialized",
        }

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """
        Record results from one training epoch.
        """

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "metrics": metrics or {},
        }

        self.history.append(record)

    def log_metadata(
        self,
        metadata: dict[str, Any],
    ) -> None:
        """
        Add metadata associated with the experiment.
        """

        self.metadata.update(metadata)

    def mark_completed(self) -> None:
        """
        Mark the experiment as completed.
        """

        self.metadata["status"] = "completed"

    def mark_failed(self, error: str) -> None:
        """
        Mark the experiment as failed.
        """

        self.metadata["status"] = "failed"
        self.metadata["error"] = error

    def save(self) -> None:
        """
        Save experiment metadata and training history.
        """

        self.experiment_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata_path = self.experiment_dir / "metadata.json"

        history_path = self.experiment_dir / "history.json"

        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.metadata,
                file,
                indent=4,
            )

        with open(
            history_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.history,
                file,
                indent=4,
            )