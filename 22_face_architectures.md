---
title: "Chapter 22 — Face Recognition Architectures"
---

[← Back to Table of Contents](./README.md)

# Chapter 22 — Face Recognition Architectures

> *"Face architectures must be fast enough for real-time, accurate enough for anti-spoofing, and discriminative enough to tell 7 billion faces apart."*

## MTCNN: Cascaded Multi-Task Face Detection (2016)

> **Planned content:** Three-stage cascade for face detection + 5-point landmark localization. P-Net (Proposal Net): stride-2 sliding window FCN → fast face/non-face + coarse bbox; runs on full image. R-Net (Refine Net): 24×24 input → refine candidates, reduce FP. O-Net (Output Net): 48×48 input → precise localization + 5 landmarks. Multi-task loss: classification + regression + landmark.

> **📊 Planned diagram:** MTCNN cascade — image pyramid → P-Net filter → R-Net refine → O-Net final landmarks. FP reduction at each stage, with box count from 10K → 100 → 5.

## RetinaFace: Multi-Task Face Detection (2019)

> **Planned content:** FPN backbone (MobileNet or ResNet) + multi-task head: face classification, 4D bbox regression, 10D 5-point landmark regression, and 3D face reconstruction (optional). SSH context module for enhanced receptive field. State-of-art face detection backbone for pipelines.

> **📊 Planned diagram:** RetinaFace head architecture — per anchor: 2 class logits + 4 box deltas + 10 landmark offsets + (optional) 3D shape coefficients.

## SCRFD: Sample-Constrained Face Detection (2021)

> **Planned content:** Efficient face detector. Sample constraint: limit positive/negative sample ratio during training. Competitive with RetinaFace at lower FLOPs. Good for mobile deployment.

## MobileFaceNet: The Efficient FR Backbone (2018)

> **Planned content:** Designed specifically for face recognition on mobile. Key differences from general MobileNetV2: bottleneck ratio reduced, global depthwise conv layer replacing GAP, Linear bottleneck to preserve face embedding quality. Input `[B, 3, 112, 112]` → 20 layers → `[B, 512]`. ~1M params, ~0.22 GFLOPs.

> **📊 Planned diagram:** MobileFaceNet architecture — block diagram with each layer's output shape `[B, C, H, W]`. Global DW conv layer highlighted.

> **📊 Planned table:** MobileFaceNet vs. IResNet-50 vs. IResNet-100 — parameters, FLOPs, LFW acc, IJB-C TAR@FAR=1e-4.

## IResNet: Improved ResNet for Face

> **Planned content:** Residual networks adapted for face recognition. Remove max pooling, use stride-1 in early layers to preserve spatial detail. IR-SE (with SE blocks). IResNet-50/100. PartialFC trick: random sample a subset of classes per mini-batch for memory-efficient large-scale training.

## FaceNet: Triplet Loss Approach (2015)

> **Planned content:** Google's approach. Inception backbone. L2-normalized 128D embedding. Triplet loss with online hard mining. 200M face training data. First to achieve near-human LFW accuracy.

## ArcFace Network Design (2019)

> **Planned content:** IResNet-50 backbone + ArcFace head. Input `[B, 3, 112, 112]` → backbone → `[B, 512]` → L2 normalize → ArcFace head `[512, C_identities]`. Training on MS1MV3 (93K identities). The complete training recipe: loss, optimizer, batch size, margin, scale.

> **📊 Planned diagram:** Full ArcFace model — backbone producing embedding, then ArcFace head transforming cosine similarities into margined logits for CE loss.

## AdaFace: Quality-Adaptive Margin (2022)

> **Planned content:** Improvement over ArcFace for low-quality (blurry, occluded) faces. Embedding norm as image quality proxy. High norm → large margin (similar to ArcFace). Low norm → small margin. Achieves better in-the-wild recognition.

> **📊 Planned diagram:** AdaFace margin function — margin vs. embedding norm curve, showing adaptive behavior.

## Model Selection Guide

> **📊 Planned table:** Face recognition model comparison — detector, backbone, landmark model, embedding size, LFW %, IJB-C TAR@FAR=1e-4, FPS on mobile.

**Next: [Chapter 23 — Tracking Architectures →](./23_tracking_architectures.md)**

---
*Last updated: May 2026*
