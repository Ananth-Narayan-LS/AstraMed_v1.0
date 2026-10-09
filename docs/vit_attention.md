
# AstraMed — ViT-B/16 Attention Visualization

## Objective

Extract and visualize attention weights from the trained ViT-B/16
binary pneumonia-classification model.

## Implementation

- `src/explainability/vit_attention.py`
  - Captures attention weights from `nn.MultiheadAttention` layers.
  - Restores original forward methods when capture is closed.
  - Converts final-layer CLS-to-patch attention into a 2D map.

- `scripts/test_vit_attention.py`
  - Loads the trained ViT-B/16 checkpoint.
  - Uses the existing DICOM preprocessing and advanced-input adapter.
  - Saves an attention map and an original-image overlay.

## Outputs

Generated files:

- `outputs/explainability/vit_attention/attention_map.png`
- `outputs/explainability/vit_attention/attention_overlay.png`

## Interpretation

The final transformer layer's attention weights are averaged across
attention heads. The CLS token's attention to patch tokens is extracted
and reshaped into a spatial grid.

For a standard ViT-B/16 with a 224x224 input, the image is divided into
14x14 patches, giving 196 patch tokens plus one CLS token.

The 14x14 spatial map is interpolated for visualization. The displayed
map is therefore a coarse, smoothed representation of patch-level
attention, not a pixel-level explanation.

## Limitations

1. Attention weights are not proof of causal feature importance.
2. The visualization uses the final transformer layer and averages
   attention across heads, potentially hiding head-specific patterns.
3. Interpolation does not add spatial detail that the model never computed.
4. The method assumes a standard CLS-token ViT architecture and square
   patch grid. Other architectures may require different extraction logic.
5. The method depends on the model exposing `nn.MultiheadAttention`.
   Implementations using fused or custom attention kernels may not expose
   weights through this interface.
6. Attention visualization alone does not establish clinical validity.
7. The model's prediction may be incorrect even when the map looks plausible.

## Execution

From the project root:

`python -u scripts/test_vit_attention.py`

The script requires the dataset and a trained ViT-B/16 checkpoint.
