---
title: "Chapter 38 — Evaluation and Benchmarks"
---

[← Back to Table of Contents](./README.md)

# Chapter 38 — Evaluation and Benchmarks

> *"Metrics are hypotheses about what matters. Choosing the wrong metric means optimizing for the wrong thing."*

## Classification Metrics

> **Planned content:** Top-1 accuracy. Top-5 accuracy. Per-class accuracy. Macro vs. micro average. Confusion matrix analysis. Precision, recall, F1 per class. Matthews Correlation Coefficient (MCC) for imbalanced. Calibration: Expected Calibration Error (ECE), reliability diagrams. ROC-AUC for binary classification.

> **📊 Planned diagram:** Confusion matrix visualization — N-class heatmap with row=GT, col=predicted, diagonal=correct, off-diagonal=confusion patterns.

## Detection Metrics: mAP Deep Dive

> **Planned content:** Step-by-step mAP computation: (1) For each class: collect all predictions across all images, sort by confidence. (2) For each threshold, compute TP/FP (IoU > threshold = TP). (3) Compute precision-recall curve. (4) Compute AP = AUC. (5) Average AP over all classes → mAP. COCO AP: average AP over IoU thresholds [0.50, 0.55, ..., 0.95]. AP50, AP75. AP_S/M/L for object size bins.

> **📊 Planned diagram (flowchart):** mAP computation step-by-step — from predictions + GT to final mAP number, with a concrete numerical example.

> **📊 Planned diagram:** Precision-Recall curve — showing how the curve changes under different score thresholds, with AP = area shaded.

```python
def compute_ap(recalls, precisions):
    """
    Compute Average Precision using all-points interpolation.
    recalls:    [N]  — recall values, sorted ascending
    precisions: [N]  — precision values at each recall
    Returns: ap (float)
    """
    # Append boundary values
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Make precision monotonically decreasing
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Find recall change points
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Sum areas of rectangles
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
```

## Segmentation Metrics

> **Planned content:** mIoU: mean over classes of IoU. Frequency-weighted IoU. Pixel accuracy. Boundary IoU for thin structures. Panoptic Quality PQ = SQ × RQ. Per-class IoU analysis. Computing mIoU efficiently in PyTorch with confusion matrix.

> **📊 Planned diagram:** IoU computation — predicted mask ∩ GT mask / predicted mask ∪ GT mask, with Venn diagram.

## Pose Estimation Metrics

> **Planned content:** OKS (Object Keypoint Similarity): Gaussian-weighted distance per keypoint. mAP at OKS thresholds [0.50:0.95]. PCK (Percentage of Correct Keypoints): fraction within distance threshold. MPJPE (Mean Per-Joint Position Error) for 3D pose in mm.

## Face Recognition Metrics

> **Planned content:** LFW protocol: pair matching accuracy, mean ± std over 10-fold. TAR@FAR: True Accept Rate at specified False Accept Rate (1e-4, 1e-3). ROC curve. Rank-1 / Rank-5 identification accuracy. EER (Equal Error Rate). DIR@FAR for open-set.

> **📊 Planned diagram:** Face verification ROC — TAR vs. FAR with operating points labeled.

## Tracking Metrics: MOTA, IDF1, HOTA

> **Planned content:** MOTA = 1 - (FN + FP + IDSW) / GT. Tracks both detection quality and identity preservation. IDF1: identity-based F1. HOTA = √(DetA × AssA) — balanced detection and association. MT (Mostly Tracked), PT, ML fragments.

## Depth Estimation Metrics

> **Planned content:** AbsRel = mean(|d_pred - d_gt| / d_gt). SqRel. RMSE. RMSE log. δ1 = fraction with max(d_pred/d_gt, d_gt/d_pred) < 1.25. SILog = scale-invariant error.

## Generative Metrics

> **Planned content:** FID (Fréchet Inception Distance): distance between feature distributions of real and generated images. IS (Inception Score): quality + diversity via KL divergence. LPIPS (Learned Perceptual Image Patch Similarity): perceptual similarity. PSNR, SSIM for super-resolution. Human evaluation / ELO for generative models.

> **📊 Planned diagram:** FID computation — extract InceptionV3 features for real and generated sets → compute Fréchet distance between multivariate Gaussians.

## FLOP Counting and Efficiency Benchmarks

> **Planned content:** FLOPs per model. Latency on standardized hardware (V100, T4, A100). Throughput (images/sec). Memory (model size + activation memory). Energy (Joules/image). Speed-accuracy Pareto curves. COCO AP vs. latency plot for detectors.

> **📊 Planned diagram:** COCO AP vs. latency scatter — YOLO series, RT-DETR, Faster-RCNN, DETR all plotted, Pareto frontier highlighted.

**Next: [Chapter 39 — Vision–Language Models →](./39_vision_language_models.md)**

---
*Last updated: May 2026*
