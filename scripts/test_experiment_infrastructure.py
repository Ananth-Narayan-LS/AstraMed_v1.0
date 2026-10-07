

from src.experiments.config import ExperimentConfig, get_device
from src.experiments.tracker import ExperimentTracker
from src.utils.reproducibility import set_seed


set_seed(42)

config = ExperimentConfig(
    model_name="resnet50",
    pretrained=True,
    dataset_image_size=512,
    model_image_size=224,
    batch_size=8,
    learning_rate=0.001,
    epochs=5,
    seed=42,
)

config.create_directories()

print("Experiment name:")
print(config.experiment_name)

print("\nExperiment directory:")
print(config.experiment_dir)

print("\nDevice:")
print(get_device())

tracker = ExperimentTracker(
    config.experiment_dir
)

tracker.log_metadata({
    "model_name": config.model_name,
    "pretrained": config.pretrained,
    "dataset_image_size": config.dataset_image_size,
    "model_image_size": config.model_image_size,
    "use_imagenet_normalization": (
        config.use_imagenet_normalization
    ),
    "batch_size": config.batch_size,
    "learning_rate": config.learning_rate,
    "epochs": config.epochs,
    "seed": config.seed,
    "device": str(get_device()),
})

tracker.log_epoch(
    epoch=1,
    train_loss=0.5329,
    val_loss=0.5186,
    metrics={
        "roc_auc": 0.7269,
    },
)

tracker.mark_completed()

tracker.save()

print("\nExperiment infrastructure test completed.")