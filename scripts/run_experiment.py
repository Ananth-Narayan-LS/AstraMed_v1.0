
"""Run a small end-to-end AstraMed experiment smoke test."""

from src.experiments.config import ExperimentConfig
from src.experiments.runner import run_experiment


def main():
    config = ExperimentConfig(
        model_name="efficientnet_b0",
        pretrained=False,  # Avoid downloading weights for this smoke test.
        dataset_image_size=512,
        model_image_size=224,
        use_imagenet_normalization=True,
        batch_size=2,
        learning_rate=0.001,
        epochs=1,
        seed=42,
    )

    result = run_experiment(
        config,
        max_train_samples=8,
        max_val_samples=4,
    )

    print("\nSMOKE TEST SUMMARY")
    print("------------------")
    print("Model:", result["model_name"])
    print("Training time:", result["training_time_seconds"])
    print("Peak GPU memory:", result["peak_gpu_memory_mb"])
    print("Best validation loss:", result["best_validation_loss"])
    print("Checkpoint:", result["best_checkpoint"])


if __name__ == "__main__":
    main()
