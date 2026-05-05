---
title: "Chapter 11 — Video Understanding"
---

[← Back to Table of Contents](./README.md)

# Chapter 11 — Video Understanding

> *"A video is a 5D tensor — and the extra dimension of time is where all the interesting semantics live."*

## Video as a Tensor

> **Planned content:** Video tensor: `[B, T, C, H, W]` or `[B, C, T, H, W]`. Storage formats (mp4, avi, frames on disk). Temporal sampling strategies (uniform, dense, stride). Frame rate and duration. Memory constraints: a 10-second 1080p clip at 30fps is ~1.5GB raw.

> **📊 Planned diagram:** Video tensor decomposition — temporal slices, showing `[T, C, H, W]` structure with time as the 4th axis.

## Action Recognition

> **Planned content:** Classify a video clip into one of K action classes. Trimmed vs. untrimmed video. Two-stream networks: RGB stream + optical flow stream → late fusion. 3D CNNs (C3D, I3D): convolutions across time and space. SlowFast: slow pathway (low frame rate, full spatial) + fast pathway (high frame rate, small spatial).

> **📊 Planned diagram:** Two-stream network architecture — RGB stream and flow stream with late fusion.

> **📊 Planned diagram:** SlowFast network — slow pathway `[B, C, T/8, H, W]` and fast pathway `[B, C/8, T, H, W]`, lateral connections.

## Temporal Modeling

> **Planned content:** 3D convolutions: extend spatial kernel to 3D `[t, h, w]`. Pseudo-3D (P3D): factorize into 2D spatial + 1D temporal. TSM (Temporal Shift Module): shift feature maps along time axis with zero extra parameters. Transformers for video: TimeSformer (divided space-time attention), Video Swin Transformer.

> **📊 Planned diagram:** TSM — feature shift illustration showing how temporal information is exchanged between frames at zero computational cost.

## Temporal Action Localization

> **Planned content:** Given an untrimmed video, predict when (start/end time) and what (class) each action occurs. Two-stage (propose + classify) vs. one-stage. BSN, BMN, AFSD. Temporal IoU (tIoU). ActivityNet benchmark.

## Video Object Segmentation (VOS)

> **Planned content:** Semi-supervised VOS: given first-frame mask, propagate to all frames. Unsupervised VOS: find salient moving objects. DAVIS benchmark. Space-time memory networks. SAM 2 for video.

## Key Datasets

> **Planned content:** Kinetics-400/600/700, Something-Something v2 (temporal reasoning), UCF-101, HMDB-51, AVA (spatio-temporal action detection), ActivityNet (temporal localization), DAVIS (VOS).

**Next: [Chapter 12 — 3D Vision →](./12_3d_vision.md)**

---
*Last updated: May 2026*
