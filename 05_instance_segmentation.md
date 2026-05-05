---
title: "Chapter 5 — Instance Segmentation"
---

[← Back to Table of Contents](./README.md)

# Chapter 5 — Instance Segmentation

> *"Semantic segmentation asks 'what is here?' — instance segmentation asks 'what and which one?'"*

## Instance vs. Semantic Segmentation

> **Planned content:** The key difference: distinguishing individual object instances. Three persons → three separate masks (instance) vs. one "person" class region (semantic). Output format: list of `(mask [H, W], class, score)` tuples per image. Challenge: variable number of instances, occlusion, overlap.

> **📊 Planned diagram:** Comparison of semantic, instance, and panoptic segmentation on the same crowded scene.

## Two-Stage Approach: Mask R-CNN

> **Planned content:** Extend Faster R-CNN with a mask head. Three parallel heads per RoI: classification, box regression, mask prediction. RoIAlign (vs. RoIPool) — why bilinear interpolation of features is critical for masks. Mask head: FCN predicting `[num_classes, 28, 28]` binary masks. During inference: apply mask of predicted class only.

> **📊 Planned diagram:** Mask R-CNN pipeline — image → FPN backbone → RPN proposals → RoIAlign → classification/box/mask heads → decoded instances.

> **📊 Planned diagram:** RoIAlign vs. RoIPool — showing the quantization error in RoIPool and how RoIAlign fixes it with bilinear interpolation.

```python
# Mask R-CNN output structure (torchvision)
# outputs: List[Dict], one dict per image
# dict keys: "boxes" [N, 4], "labels" [N], "scores" [N], "masks" [N, 1, H, W]

for i, pred in enumerate(outputs):
    boxes  = pred["boxes"]    # [N, 4]  — xyxy format
    labels = pred["labels"]   # [N]     — class indices
    scores = pred["scores"]   # [N]     — confidence
    masks  = pred["masks"]    # [N, 1, H, W]  — soft masks (0-1)
    binary = (masks > 0.5).squeeze(1)  # [N, H, W]  — binary masks
```

## One-Stage Approaches

> **Planned content:** YOLACT: prototype masks + mask coefficients. CondInst: dynamic filter generation conditioned on instance features. SOLOv2: predict mask at each grid cell. Advantage: faster than two-stage, single forward pass.

> **📊 Planned diagram:** YOLACT pipeline — prototype generation + per-instance coefficient prediction → linear combination → mask.

## Query-Based Approaches

> **Planned content:** Mask2Former: object queries + masked attention. Each query predicts one instance. Unified panoptic model. QueryInst. How masked cross-attention restricts each query to attend only to its predicted region.

> **📊 Planned diagram:** Mask2Former architecture — pixel decoder + transformer decoder with masked attention, showing query-to-instance assignment.

## Panoptic Segmentation — Unified View

> **Planned content:** Combining things (instances) and stuff (amorphous regions). Panoptic Quality metric: PQ = SQ × RQ. Panoptic FPN, Panoptic DeepLab. UPSNet. OneFormer — single model with task-conditioned queries for semantic, instance, and panoptic.

> **📊 Planned diagram:** PQ computation — matched segments (SQ) × correctly detected segments (RQ).

## Segment Anything Model (SAM)

> **Planned content:** SAM architecture: image encoder (ViT-H), prompt encoder (points, boxes, masks, text), mask decoder (lightweight transformer). Promptable segmentation: any point/box/text → mask. SA-1B dataset: 1 billion masks. SAM 2 for video. Zero-shot segmentation.

> **📊 Planned diagram:** SAM architecture — image encoder produces image embedding `[B, 256, 64, 64]`, prompt encoder produces sparse/dense embeddings, mask decoder outputs 3 masks + IoU scores.

> **📊 Planned diagram:** SAM prompting modes — point prompt, box prompt, text prompt (SAM 2), everything mode.

```python
from segment_anything import sam_model_registry, SamPredictor

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)

predictor.set_image(image_rgb)  # encodes image: [H, W, 3] → [1, 256, 64, 64]

# Point prompt: [[x, y]] coordinates
masks, scores, logits = predictor.predict(
    point_coords=np.array([[500, 375]]),  # [1, 2]
    point_labels=np.array([1]),           # [1] — 1=foreground
    multimask_output=True
)
# masks: [3, H, W]  — 3 mask candidates
# scores: [3]       — IoU predictions per mask
```

## Evaluation Metrics

> **Planned content:** AP for instance segmentation — same framework as detection but IoU on masks. Mask IoU computation. COCO AP@[0.5:0.95] for instance segmentation. Boundary IoU. Panoptic Quality (PQ), Semantic Quality (SQ), Recognition Quality (RQ).

**Next: [Chapter 6 — Object Tracking →](./06_object_tracking.md)**

---
*Last updated: May 2026*
