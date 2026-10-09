
"""Run a consistent screening benchmark for AstraMed models."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.config import ExperimentConfig
from src.experiments.runner import run_experiment

MODELS = [
    "resnet50",
    "densenet121",
    "efficientnet_b0",
    "convnext_tiny",
    "vit_b16",
]


def main():
    print("=" * 65)
    print("ASTRAMED MODEL SCREENING BENCHMARK")
    print("=" * 65)
    print("Pretrained models | 2 epochs | 1000 training / 300 validation samples")
    print("These are screening experiments, not final clinical performance results.")

    for model_name in MODELS:
        print(f"\n{'=' * 65}")
        print(f"Starting: {model_name}")
        print("=" * 65)

        config = ExperimentConfig(
            model_name=model_name,
            pretrained=True,
            dataset_image_size=512,
            model_image_size=224,
            use_imagenet_normalization=True,
            batch_size=4,
            learning_rate=0.0001,
            epochs=2,
            seed=42,
        )

        try:
            result = run_experiment(
                config,
                max_train_samples=1000,
                max_val_samples=300,
            )
            print(f"Finished: {model_name}")
            print(result)
        except Exception as error:
            print(f"FAILED: {model_name}")
            print(f"Reason: {error}")
            print("Continuing to the next model.")

    print("\nScreening benchmark loop finished.")
    print("Check each experiment's status before comparing results.")
    print("Next, run: python scripts/compare_models.py")


if __name__ == "__main__":
    main()
