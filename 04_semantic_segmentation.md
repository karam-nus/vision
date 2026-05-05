---
title: "Chapter 4 — Semantic Segmentation"
---

[← Back to Table of Contents](./README.md)

# Chapter 4 — Semantic Segmentation

> *"Where detection draws boxes around things, segmentation traces their exact boundaries — pixel by pixel."*

## The Segmentation Problem

> **Planned content:** Assign a semantic class label to every pixel. Output: `[B, H, W]` class map (dense prediction). Difference from detection (no instance distinction). Difference from instance segmentation. Applications: autonomous driving (scene parsing), medical imaging, satellite imagery, robotics.

> **📊 Planned diagram:** Semantic segmentation output — original image → per-pixel class overlay, with class color legend.

## Encoder-Decoder Architecture

> **Planned content:** Why a pure classification backbone loses spatial resolution (stride 32). The encoder-decoder design: encoder (backbone) → bottleneck → decoder (upsample back to full resolution). Skip connections: why they are critical for recovering fine spatial details. The tension between semantic depth (deep encoder) and spatial precision (decoder).

> **📊 Planned diagram:** Encoder-decoder architecture with tensor shapes at each stage — `[B, C, H/32, W/32]` up to `[B, num_classes, H, W]` after decoder.

## Dilated (Atrous) Convolutions

> **Planned content:** Standard stride-based downsampling loses spatial resolution. Dilated convolutions: insert zeros between filter weights to increase receptive field without reducing spatial size. Dilation rate r. Effective receptive field calculation. Gridding artifacts and how to fix them (hybrid dilated convolutions).

> **📊 Planned diagram:** Dilated convolution with rates 1, 2, 4 — showing how receptive field grows without spatial downsampling. Feature map size stays constant.

$$y[i] = \sum_{k} x[i + r \cdot k] \cdot w[k], \quad \text{dilation rate } r$$

## Atrous Spatial Pyramid Pooling (ASPP)

> **Planned content:** Capturing objects at multiple scales without multi-scale inference. ASPP: parallel dilated convolutions with rates 6, 12, 18 + global average pooling + 1×1 conv. Concatenate and fuse. Introduced in DeepLab v2, improved in v3. Light-ASPP for efficiency.

> **📊 Planned diagram:** ASPP module — parallel branches with different dilation rates, concatenation, and output. Tensor shapes: input `[B, C, H/8, W/8]` → output `[B, 256, H/8, W/8]`.

## Skip Connections and Multi-Scale Fusion

> **Planned content:** Low-level features (edges, textures) from early encoder layers. High-level features (semantics) from deep encoder. How to combine them: addition vs. concatenation. Feature Pyramid Networks (FPN) for segmentation. Deep supervision. ASPP + decoder (DeepLab v3+).

> **📊 Planned diagram:** Skip connection fusion — showing how encoder features at stride 4 and stride 8 are fused with decoder output.

## Segmentation Loss Functions

> **Planned content:** Pixel-wise cross-entropy: simple but imbalanced. Dice loss: overlap-based, good for class imbalance. Lovász-Softmax loss: surrogate for mIoU. Tversky loss: adjustable precision-recall balance. Boundary loss: emphasizes boundary pixels. Combined losses in practice.

> **📊 Planned diagram:** Class imbalance visualization — background pixels (95%) vs. foreground (5%), motivating dice/lovász.

$$\text{Dice Loss} = 1 - \frac{2 \sum p_i g_i}{\sum p_i + \sum g_i}$$

## Evaluation: mIoU

> **Planned content:** Mean Intersection over Union. Per-class IoU calculation. Why mIoU is better than pixel accuracy for imbalanced datasets. Frequency-weighted IoU. Boundary IoU (for thin structures). Computing mIoU efficiently in PyTorch.

> **📊 Planned diagram (flowchart):** mIoU computation — predicted mask vs. GT mask → per-class TP/FP/FN → per-class IoU → mean over classes.

## Panoptic Segmentation

> **Planned content:** Unified task combining semantic (stuff: sky, road) and instance (things: person, car) segmentation. Panoptic Quality (PQ) = SQ × RQ. Panoptic FPN. Unified labeling with segment IDs. Why panoptic is the complete scene understanding task.

> **📊 Planned diagram:** Semantic vs. instance vs. panoptic segmentation comparison — same image with three different output representations.

## Practical Code: Semantic Segmentation Inference

> **Planned content:** Running DeepLab v3+ and SegFormer from HuggingFace / timm on a custom image. Visualizing predictions. Computing mIoU on a validation set.

```python
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import torch

processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")

inputs = processor(images=image_pil, return_tensors="pt")
# inputs["pixel_values"]: [B=1, C=3, H=512, W=512]

with torch.no_grad():
    outputs = model(**inputs)
    # outputs.logits: [B=1, num_labels=150, H/4=128, W/4=128]

logits_upsampled = torch.nn.functional.interpolate(
    outputs.logits, size=image_pil.size[::-1], mode="bilinear", align_corners=False
)  # [B=1, 150, H_orig, W_orig]
pred_seg = logits_upsampled.argmax(dim=1)  # [B=1, H_orig, W_orig]
```

**Next: [Chapter 5 — Instance Segmentation →](./05_instance_segmentation.md)**

---
*Last updated: May 2026*
