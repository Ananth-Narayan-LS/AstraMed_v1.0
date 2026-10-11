"""The experiment runner must refuse to overwrite an existing checkpoint."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.config import ExperimentConfig
from src.experiments.runner import run_experiment


def test_existing_checkpoint_is_not_overwritten():
    with tempfile.TemporaryDirectory() as tmp:
        config = ExperimentConfig(model_name="resnet50", pretrained=False, output_dir=Path(tmp))
        config.checkpoint_dir.mkdir(parents=True)
        checkpoint = config.checkpoint_dir / "best_model.pth"
        checkpoint.write_bytes(b"sentinel")

        try:
            run_experiment(config, max_train_samples=2, max_val_samples=2)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Runner overwrote or ignored an existing checkpoint.")

        assert checkpoint.read_bytes() == b"sentinel"


if __name__ == "__main__":
    test_existing_checkpoint_is_not_overwritten()
    print("PASS: test_existing_checkpoint_is_not_overwritten")
