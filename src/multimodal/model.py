"""
Multimodal pneumonia model: image encoder + structured clinical encoder (+ optional text).

Modality mask columns: [image, structured, symptoms_text, history_text].
A missing modality contributes an exactly-zero embedding and its mask bit is passed to
the fusion head, so absence is explicit and never looks like clinical evidence.

The image encoder is a torchvision ResNet50 with the same layer names as the frozen
image-only baseline, so baseline weights load into it (everything except the old
1-logit `fc`). The baseline checkpoint file is only read, never written.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from src.data.advanced_input import AdvancedInputAdapter
from src.multimodal.text import ClinicalTextEncoder

MASK_COLUMNS = ("image", "structured", "symptoms", "history")
IMAGE_FEATURE_DIM = 2048


class MultimodalPneumoniaNet(nn.Module):
    def __init__(
        self,
        struct_dim: int = 0,
        use_structured: bool = False,
        use_text: bool = False,
        vocab_size: int = 0,
        embed_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        if use_structured and struct_dim <= 0:
            raise ValueError("struct_dim must be positive when use_structured=True.")
        if use_text and vocab_size <= 0:
            raise ValueError("vocab_size must be positive when use_text=True.")
        self.config = {
            "struct_dim": struct_dim, "use_structured": use_structured,
            "use_text": use_text, "vocab_size": vocab_size,
            "embed_dim": embed_dim, "dropout": dropout,
        }
        self.use_structured, self.use_text = use_structured, use_text

        self.input_adapter = AdvancedInputAdapter(image_size=224, use_imagenet_normalization=True)
        self.image_encoder = models.resnet50(weights=None)
        self.image_encoder.fc = nn.Identity()
        self.image_projection = nn.Sequential(nn.Linear(IMAGE_FEATURE_DIM, embed_dim), nn.ReLU())

        branches = 1
        self.active_mask_columns = [0]
        if use_structured:
            self.structured_encoder = nn.Sequential(
                nn.Linear(struct_dim, 64), nn.ReLU(), nn.Linear(64, embed_dim), nn.ReLU()
            )
            branches += 1
            self.active_mask_columns.append(1)
        if use_text:
            self.text_encoder = ClinicalTextEncoder(vocab_size, out_dim=embed_dim)
            branches += 2  # symptoms and history share the encoder
            self.active_mask_columns += [2, 3]

        self.fusion_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim * branches + len(self.active_mask_columns), 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    # ------------------------------------------------------------------ baseline weights
    def load_image_encoder_from_baseline(self, checkpoint_path) -> dict:
        """
        Copy the baseline ResNet50 weights into the image encoder. The checkpoint is read
        with weights_only=True and never modified. Returns a provenance report.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = checkpoint["model_state_dict"]
        incompatible = self.image_encoder.load_state_dict(state, strict=False)
        expected_unused = {"fc.weight", "fc.bias"}
        if incompatible.missing_keys or set(incompatible.unexpected_keys) != expected_unused:
            raise RuntimeError(
                "Baseline weights do not match the image encoder: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        return {
            "initialized_from": str(checkpoint_path),
            "baseline_epoch": checkpoint.get("epoch"),
            "tensors_loaded": len(state) - len(expected_unused),
            "tensors_not_loaded": sorted(expected_unused),
            "note": "Baseline 1-logit classifier head is discarded; projection and fusion head are new.",
        }

    def set_image_encoder_trainable(self, trainable: bool) -> None:
        for parameter in self.image_encoder.parameters():
            parameter.requires_grad = trainable

    # ------------------------------------------------------------------ forward
    def forward(self, batch: dict) -> torch.Tensor:
        """Return logits with shape [B, 1]."""
        image = batch["image"]
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError(f"Expected image [B, 1, H, W], got {tuple(image.shape)}.")
        mask = batch["modality_mask"].float()
        if mask.shape != (image.shape[0], len(MASK_COLUMNS)):
            raise ValueError(
                f"modality_mask must be [B, {len(MASK_COLUMNS)}], got {tuple(mask.shape)}."
            )

        parts = [self.image_projection(self.image_encoder(self.input_adapter(image)))]

        if self.use_structured:
            features = batch["clinical_features"]
            if features.shape != (image.shape[0], self.config["struct_dim"]):
                raise ValueError(f"clinical_features has shape {tuple(features.shape)}.")
            parts.append(self.structured_encoder(features) * mask[:, 1:2])
        if self.use_text:
            for column, prefix in ((2, "symptom"), (3, "history")):
                encoded = self.text_encoder(batch[f"{prefix}_ids"], batch[f"{prefix}_mask"])
                parts.append(encoded * mask[:, column : column + 1])

        parts.append(mask[:, self.active_mask_columns])
        return self.fusion_head(torch.cat(parts, dim=1))

    @torch.no_grad()
    def predict_proba(self, batch: dict) -> torch.Tensor:
        """Probabilities [B] of pneumonia (sigmoid of the logit)."""
        was_training = self.training
        self.eval()
        try:
            return torch.sigmoid(self(batch)).reshape(-1)
        finally:
            self.train(was_training)


def save_multimodal_checkpoint(model, optimizer, epoch, val_metric, path, extra: dict) -> None:
    from pathlib import Path

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists; refusing to overwrite it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "astramed_multimodal",
            "epoch": epoch,
            "val_metric": val_metric,
            "model_config": model.config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "extra": extra,
        },
        path,
    )


def load_multimodal_checkpoint(path, device="cpu") -> tuple["MultimodalPneumoniaNet", dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("kind") != "astramed_multimodal":
        raise ValueError("Not a multimodal checkpoint (refusing to load it as one).")
    model = MultimodalPneumoniaNet(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device), checkpoint
