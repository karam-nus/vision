---
title: "Chapter 3 — Object Detection"
---

[← Back to Table of Contents](./README.md)

# Chapter 3 — Object Detection

> *"Detection is classification with a ruler — the hard part is the ruler."*

## The Detection Problem

> **Planned content:** Formal definition: given an image, output a set of bounding boxes and class labels. Why it is harder than classification: variable number of outputs, spatial localization, scale variation. Output format: `[x_min, y_min, x_max, y_max, class, score]` per box. Common box formats: `[x1, y1, x2, y2]` (xyxy), `[cx, cy, w, h]` (cxcywh), `[x, y, w, h]` (xywh).

> **📊 Planned diagram (flowchart):** Detection output decomposition — image → detector → list of (box, class, confidence) tuples, with coordinate system illustration.

## Two-Stage vs. One-Stage Detectors

> **Planned content:** Two-stage (R-CNN family): first propose regions, then classify + refine. Accurate but slower. One-stage (YOLO, SSD, FCOS): predict boxes and classes in a single pass. Faster, suitable for real-time. The accuracy-speed tradeoff. Modern convergence (RT-DETR, YOLOv9+).

> **📊 Planned diagram:** Two-stage pipeline (Faster R-CNN) vs. one-stage pipeline (YOLO) — architectural overview with tensor shapes.

## Anchor-Based Detection

> **Planned content:** What are anchors? Fixed reference boxes tiled across the feature map. Anchor scales and aspect ratios. Anchor assignment: positive (IoU > threshold), negative (IoU < threshold), ignored. Regression targets (delta encoding from anchor to ground truth). How anchor design affects performance.

> **📊 Planned diagram:** Anchor tiling over feature map — showing anchors at different scales/ratios, with IoU thresholding visualization.

$$t_x = \frac{x_{gt} - x_a}{w_a}, \quad t_y = \frac{y_{gt} - y_a}{h_a}, \quad t_w = \log\frac{w_{gt}}{w_a}, \quad t_h = \log\frac{h_{gt}}{h_a}$$

## Anchor-Free Detection

> **Planned content:** FCOS: predict box from each foreground pixel. CenterNet: detect object centers + offsets + size. Key insight: anchor-free methods avoid hyperparameter-sensitive anchor design. Label assignment challenges without anchors (e.g., ambiguity at object boundaries).

> **📊 Planned diagram:** FCOS centerness map and offset regression — how each pixel predicts a box.

## IoU and Its Variants — A Deep Dive

> **Planned content:** Intersection over Union: the fundamental metric. Standard IoU formula. Why IoU = 0 gives no gradient when boxes don't overlap. GIoU (generalized IoU) — add enclosing box term. DIoU — add center distance term. CIoU — add aspect ratio consistency. SIoU — add angle term. EIoU — explicit width/height. When to use each.

> **📊 Planned diagram:** Visual comparison of IoU, GIoU, DIoU, CIoU — showing what each loss term captures geometrically.

$$\text{IoU} = \frac{|A \cap B|}{|A \cup B|}, \qquad \text{GIoU} = \text{IoU} - \frac{|C \setminus (A \cup B)|}{|C|}$$

## ⭐ NMS — Non-Maximum Suppression Deep Dive

> **Planned content:** Why multiple detections appear per object. The core NMS algorithm step by step. Greedy NMS: sort by score, suppress overlapping boxes. Limitations: crowded scenes, touching objects. Soft-NMS: decay rather than eliminate. DIoU-NMS: use distance-aware IoU. WBF (Weighted Box Fusion): weighted average instead of suppression. Matrix NMS. CUDA-accelerated NMS. NMS hyperparameters (IoU threshold, score threshold) and their tuning.

> **📊 Planned diagram (flowchart):** Greedy NMS algorithm step-by-step — sort → pick top → suppress → repeat, with concrete box examples at each step.

> **📊 Planned diagram:** Failure case visualization — greedy NMS dropping true positives in crowded scenes; Soft-NMS vs. WBF comparison.

```python
def nms(boxes, scores, iou_threshold=0.5):
    """
    boxes:  [N, 4]  — xyxy format
    scores: [N]     — confidence scores
    returns: [K]    — indices of kept boxes (K ≤ N)
    """
    # Step 1: sort by descending confidence
    order = scores.argsort(descending=True)   # [N]
    keep = []

    while order.numel() > 0:
        # Step 2: always keep the highest-score box
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break

        # Step 3: compute IoU of top box with all remaining boxes
        ious = compute_iou(boxes[i], boxes[order[1:]])  # [N-1]

        # Step 4: keep only boxes below the IoU threshold
        mask = ious < iou_threshold   # [N-1] bool
        order = order[1:][mask]       # suppress overlapping boxes

    return keep
```

> **📊 Planned diagram (flowchart):** Soft-NMS algorithm — showing the score decay function (linear vs. Gaussian) vs. hard suppression.

## Label Assignment Strategies

> **Planned content:** SimOTA (YOLOX). Task-Aligned Assigner (TOOD, YOLOv8). OTA (Optimal Transport Assignment). ATSS (Adaptive Training Sample Selection). Why assignment strategy is the key difference between modern detectors. The dynamic vs. fixed assignment distinction.

> **📊 Planned diagram:** Comparison of assignment strategies — fixed IoU threshold vs. dynamic OTA/SimOTA — showing which GT is assigned to which anchor.

## Focal Loss

> **Planned content:** The class imbalance problem in one-stage detection — overwhelming easy negatives. Focal loss derivation: `FL(p_t) = -(1 - p_t)^γ log(p_t)`. γ parameter effect. α balancing. Why focal loss enabled RetinaNet to match two-stage accuracy.

> **📊 Planned diagram:** Focal loss vs. CE loss curves for different γ values — showing down-weighting of easy examples.

$$\text{FL}(p_t) = -(1-p_t)^\gamma \log(p_t)$$

## Feature Pyramid Networks (FPN)

> **Planned content:** The scale problem — small objects need high-resolution features, large objects need semantic features. FPN: top-down pathway with lateral connections. Tensor shapes through FPN. P3, P4, P5 (and P2, P6, P7) output strides. PANet, BiFPN (EfficientDet), AFPN variants.

> **📊 Planned diagram:** FPN architecture — bottom-up backbone, top-down pathway, lateral connections, with feature map sizes `[B, C, H/8, W/8]` through `[B, C, H/64, W/64]`.

## Detection Loss Functions

> **Planned content:** Multi-task loss: classification loss + regression loss (+ objectness in some designs). Balancing coefficients. SmoothL1 (Huber) vs. L1 vs. IoU-based losses. Binary cross-entropy for objectness. Class-agnostic vs. class-specific boxes.

## Evaluation: mAP Computation

> **Planned content:** Precision and recall for detection. Matching predictions to ground truth (IoU threshold). Precision-recall curve. Average Precision (AP) = area under PR curve. mAP = mean over all classes. COCO mAP: AP@[0.5:0.95] (primary), AP@50, AP@75. AP_S, AP_M, AP_L for size analysis. The 11-point interpolation vs. all-points interpolation.

> **📊 Planned diagram (flowchart):** mAP computation pipeline — predictions → sort by score → match to GT → build PR curve → compute AP → average over classes.

> **📊 Planned diagram:** Precision-Recall curve with AP = AUC highlighted. Example with multiple classes.

## Practical Code: Inference with a Pre-Trained Detector

> **Planned content:** Running YOLOv8 on an image. Parsing outputs. Visualizing boxes. Running on video. Using Ultralytics, Hugging Face transformers, and timm.

**Next: [Chapter 4 — Semantic Segmentation →](./04_semantic_segmentation.md)**

---
*Last updated: May 2026*
