---
title: "Chapter 19 — Detection Architectures"
---

[← Back to Table of Contents](./README.md)

# Chapter 19 — Detection Architectures

> *"Detection architectures encode years of hard-won engineering intuition: how to handle scale, localize without anchors, and match the right proposal to the right ground truth."*

## The R-CNN Family: Two-Stage Pioneers

> **Planned content:** R-CNN (2013): selective search proposals → warp → CNN → SVM classifier. Slow (2000 crops per image). Fast R-CNN: single forward pass, RoIPool for feature extraction. Faster R-CNN: replace selective search with Region Proposal Network (RPN). RPN shares backbone features. Cascade R-CNN: multi-stage bounding box refinement with increasing IoU thresholds.

> **📊 Planned diagram:** R-CNN family evolution — R-CNN vs. Fast R-CNN vs. Faster R-CNN, showing computation flow and speed improvement at each step.

> **📊 Planned diagram:** Faster R-CNN architecture — backbone → FPN → RPN (objectness + box) → RoIAlign → classification + regression heads. Tensor shapes throughout.

## SSD and RetinaNet

> **Planned content:** SSD: multi-scale anchors from multiple feature map levels. RetinaNet: FPN + focal loss + class/box subnets. Why focal loss enabled single-stage to match two-stage accuracy. RetinaNet as the canonical one-stage baseline.

> **📊 Planned diagram:** RetinaNet with FPN — P3 to P7 feature maps, shared class and box subnets, anchor grid visualization.

## YOLO Series: Speed-Accuracy Pareto Frontier

> **Planned content:** YOLO (v1–v11) is the most deployed detection family. Each version made key architectural or training improvements.

### YOLOv1 (2015)
> **Planned content:** Divide image into S×S grid. Each cell predicts B boxes + confidence + C class probs. Single regression output. 45 FPS. The unified detection paradigm.

> **📊 Planned diagram:** YOLOv1 grid prediction — `[S, S, B*5+C]` output tensor explained cell by cell.

### YOLOv2 (2016) — YOLO9000
> **Planned content:** Anchor boxes, batch normalization, high-resolution classifier, multi-scale training. WordTree for 9000-class detection.

### YOLOv3 (2018)
> **Planned content:** Multi-scale prediction (3 scales, like FPN). Darknet-53 backbone. Logistic classifier instead of softmax (multi-label). 320-608 input sizes. Still widely used baseline.

> **📊 Planned diagram:** YOLOv3 multi-scale output — three detection heads at strides 8, 16, 32, each predicting `[B, 3*(5+C), H/s, W/s]`.

### YOLOv4 (2020)
> **Planned content:** Bag of Freebies (training tricks) + Bag of Specials (architectural tricks). CSPNet backbone, PANet neck, Mish activation, mosaic augmentation, DropBlock.

### YOLOv5
> **Planned content:** Ultralytics implementation. YAML-configurable. P2/P3/P4/P5 model variants. Widely adopted in production. AutoAnchor. FP16 export.

### YOLOv6, YOLOv7, YOLOv8
> **Planned content:** YOLOv6 (Meituan): RepVGG blocks, decoupled head. YOLOv7: E-ELAN, COCO SOTA in 2022. YOLOv8 (Ultralytics): anchor-free, unified interface for detection/segmentation/pose/classification.

> **📊 Planned diagram:** YOLOv8 architecture — CSPDarknet backbone → C2f blocks → FPN+PAN neck → decoupled anchor-free head with DFL (Distribution Focal Loss).

### YOLOv9, YOLOv10, YOLOv11
> **Planned content:** YOLOv9: GELAN + PGI (Programmable Gradient Information). YOLOv10: NMS-free end-to-end detection. YOLOv11: Ultralytics v11 with C3k2 blocks and efficiency improvements.

> **📊 Planned table:** YOLO version comparison — year, backbone, neck, head type, anchor-free?, COCO mAP, inference speed.

## Anchor-Free Methods

> **Planned content:** FCOS: per-pixel box regression (l, r, t, b) + centerness score. CenterNet: detect objects as center points on heatmap + offset + size. CornerNet: detect corners via corner pooling + associative embedding.

> **📊 Planned diagram:** CenterNet head — three branches: heatmap `[B, C, H/4, W/4]` + offset `[B, 2, H/4, W/4]` + size `[B, 2, H/4, W/4]`. Decoding process.

## DETR and Transformer-Based Detection

> **Planned content:** DETR: remove anchors and NMS entirely. N object queries → cross-attention with image features → directly predict N boxes and classes. Bipartite matching loss (Hungarian). Slow convergence. Deformable DETR: deformable attention for faster training. DN-DETR, DAB-DETR, DINO (Detection Transformers).

> **📊 Planned diagram:** DETR architecture — backbone → positional encoding → encoder-decoder transformer → N queries → prediction heads → Hungarian matching with GT.

> **📊 Planned diagram:** Hungarian bipartite matching in DETR training — N predictions matched to M GT boxes (N >> M), showing the assignment and unmatched queries predicting "no object".

## RT-DETR: Real-Time Transformer Detection

> **Planned content:** PaddleDetection's RT-DETR: hybrid encoder (CNN + attention), efficient decoder. Matches YOLO speed with DETR accuracy. RT-DETRv2.

> **📊 Planned diagram:** RT-DETR vs. YOLO on accuracy-speed scatter plot.

## Architecture Comparison

> **📊 Planned table:** Detection architectures — anchor type, backbone, neck, NMS required, COCO AP, latency (ms) on T4.

**Next: [Chapter 20 — Segmentation Architectures →](./20_segmentation_architectures.md)**

---
*Last updated: May 2026*
