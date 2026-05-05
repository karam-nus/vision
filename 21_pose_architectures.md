---
title: "Chapter 21 — Pose Estimation Architectures"
---

[← Back to Table of Contents](./README.md)

# Chapter 21 — Pose Estimation Architectures

> *"The architecture for pose must preserve high-resolution spatial detail while capturing global context — a tension that defines the field."*

## Stacked Hourglass Networks (2016)

> **Planned content:** Stack multiple encoder-decoder "hourglass" modules sequentially. Each hourglass: downsample to global context → upsample to full resolution with skip connections. Stack 4-8 hourglasses with intermediate supervision at each. Bottom-up processing + top-down refinement iteratively.

> **📊 Planned diagram:** Stacked Hourglass — two chained hourglass modules with intermediate heatmap supervision, showing tensor flow `[B, K, 64, 64]` at each stage.

## SimpleBaseline (2018)

> **Planned content:** Deceptively simple: ResNet backbone + 3 deconvolutional (transposed conv) layers. Works surprisingly well. Lightweight, easy to train. Established strong baseline for COCO top-down.

> **📊 Planned diagram:** SimpleBaseline — ResNet → 3 deconv layers `[B, 256, 64, 64]` → 1×1 conv → heatmaps `[B, K, 64, 64]`.

## HRNet: High-Resolution Net (2019)

> **Planned content:** Key insight: don't sacrifice spatial resolution. Maintain high-resolution representations throughout the network. Multi-resolution parallel branches: HRNet simultaneously processes `[B, C, H/4, W/4]`, `[B, 2C, H/8, W/8]`, `[B, 4C, H/16, W/16]`, `[B, 8C, H/32, W/32]`. Multi-resolution fusion: exchange information between branches at each stage. Final heatmap predicted from high-resolution branch.

> **📊 Planned diagram:** HRNet multi-resolution parallel branches — 4 resolution levels maintained in parallel, with fusion modules at each stage. Contrast with downsampled-then-upsampled alternatives.

> **📊 Planned diagram:** HRNet fusion block — bilinear upsample of lower-res branches + strided conv downsample of higher-res branches → add → fused feature maps.

> **📊 Planned table:** HRNet-W32 vs. HRNet-W48 — parameters, GFLOPs, COCO keypoint AP.

## ViTPose: Vision Transformer for Pose (2022)

> **Planned content:** Plain ViT (non-hierarchical) as backbone for pose. Simple decoder (2 deconv layers). Scale up to ViT-H for SOTA. Multi-task training on multiple pose datasets. Why ViT representations transfer well to pose estimation.

> **📊 Planned diagram:** ViTPose — ViT-B/L/H patch tokens `[B, 256, 768]` → reshape to `[B, 768, 16, 16]` → deconv decoder → heatmaps `[B, K, 64, 64]`.

## OpenPose: Bottom-Up with PAFs (2017)

> **Planned content:** (PAFs covered in depth in Ch. 7) — Architecture: VGG-based two-branch network predicting heatmaps + PAFs simultaneously. Multi-stage refinement (6 stages). COCO whole-body extension. Widely deployed real-time pose.

> **📊 Planned diagram:** OpenPose architecture — shared VGG features → two branches (heatmap, PAF) × 6 stages → assembly.

## DEKR: Disentangled Keypoint Regression (2021)

> **Planned content:** Bottom-up anchor-free approach. Adaptive convolution centered on each pixel. Disentangled learning: separate feature extraction per keypoint type. Eliminates heatmap → coordinate quantization error. HRNet backbone.

## WholeBody Pose

> **Planned content:** ZoomNet: zoom into face/hand/feet regions for fine-grained detection. OSX: one-stage expressive whole-body. ExPose: part-specific feature extractors. 133-keypoint COCO WholeBody benchmark.

**Next: [Chapter 22 — Face Architectures →](./22_face_architectures.md)**

---
*Last updated: May 2026*
