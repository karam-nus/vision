---
title: "Chapter 20 — Segmentation Architectures"
---

[← Back to Table of Contents](./README.md)

# Chapter 20 — Segmentation Architectures

> *"From coarse features to pixel-perfect boundaries — the architecture is the bridge between semantic understanding and spatial precision."*

## FCN: The Founding Architecture (2014)

> **Planned content:** Fully Convolutional Network: replace FC layers with 1×1 conv for dense prediction. Skip connections from pool3, pool4 to decoder. FCN-32s, FCN-16s, FCN-8s variants — improving spatial detail with each. Why FCN was revolutionary despite rough output.

> **📊 Planned diagram:** FCN architecture — VGG backbone with FC layers replaced by 1×1 conv, bilinear upsample, skip additions. FCN-32s vs. FCN-8s comparison.

## U-Net: The Universal Segmentation Architecture (2015)

> **Planned content:** Encoder-decoder with dense skip connections. Every encoder level connects directly to corresponding decoder level. Works extremely well with small datasets. Cropping in original (no padding). U-Net++: nested skip connections. U²-Net: residual U-blocks. Attention U-Net.

> **📊 Planned diagram:** U-Net architecture — full tensor shapes at every stage, symmetric structure, skip connections visualized. Left: encoder (contracting), right: decoder (expanding).

## DeepLab Series (v1→v3+)

> **Planned content:** DeepLab v1: atrous conv + CRF post-processing. V2: multi-scale ASPP. V3: improved ASPP + global pooling. V3+: decoder with low-level features from backbone for sharper boundaries. Xception backbone. Backbone-neck-head design.

> **📊 Planned diagram:** DeepLabV3+ architecture — Xception encoder → ASPP module → decoder merging low-level features → bilinear upsample → output `[B, C, H, W]`.

## Mask R-CNN for Instance Segmentation

> **Planned content:** (Covered in Ch. 5) — architecture deep dive here: FPN backbone, RPN, RoIAlign, mask head (FCN predicting `[C, 28, 28]` per RoI). Decoupling classification and segmentation (class-agnostic mask head works best).

## Panoptic FPN

> **Planned content:** Extend Mask R-CNN with a semantic segmentation branch. Share FPN backbone between instance (Mask R-CNN) and semantic (FCN) heads. Merge predictions: instances override stuff predictions. Panoptic quality evaluation.

## Mask2Former: Universal Segmentation (2021)

> **Planned content:** Unified architecture for semantic, instance, and panoptic segmentation. Pixel decoder (multi-scale FPN features) + transformer decoder (object queries). Masked cross-attention: each query only attends to its predicted foreground region. Top-K queries. Task-specific heads on top.

> **📊 Planned diagram:** Mask2Former architecture — backbone → pixel decoder (multi-scale features) → transformer decoder with masked attention → per-query class + mask predictions.

## SAM: Segment Anything (2023)

> **Planned content:** (Architecture details covered in Ch. 5) — training at scale: SA-1B dataset (1B masks). Interactive annotation engine. Zero-shot. SAM 2 for video with streaming memory.

## OneFormer: One Transformer for All Segmentation

> **Planned content:** Task-conditioned architecture: a task token (`[semantic]`, `[instance]`, `[panoptic]`) conditions all attention layers. Single model trained jointly on all three tasks. Outperforms specialized models in some settings.

## SegFormer

> **Planned content:** Hierarchical transformer encoder (Mix Transformer, MiT-B0 to B5) + lightweight MLP decoder. No positional encoding needed — interpolation at test time. Fast and accurate. NV-NVIDIA official HuggingFace model.

> **📊 Planned diagram:** SegFormer architecture — MiT encoder producing 4 feature scales → MLP decoder concatenating and projecting → segmentation output.

## Architecture Selection Guide

> **Planned content:** When to use U-Net (medical, satellite, limited data), DeepLabV3+ (general semantic), Mask2Former (panoptic/instance), SegFormer (efficient transformer), SAM (promptable zero-shot).

> **📊 Planned table:** Segmentation architecture comparison — task support (semantic/instance/panoptic), backbone, ADE20K mIoU, COCO panoptic PQ, parameters, speed.

**Next: [Chapter 21 — Pose Architectures →](./21_pose_architectures.md)**

---
*Last updated: May 2026*
