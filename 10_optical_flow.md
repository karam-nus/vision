---
title: "Chapter 10 — Optical Flow"
---

[← Back to Table of Contents](./README.md)

# Chapter 10 — Optical Flow

> *"Optical flow is the apparent motion of brightness patterns — the visual velocity field of the world."*

## The Optical Flow Problem

> **Planned content:** Dense motion estimation: for every pixel (x, y) at time t, estimate displacement (u, v) to time t+1. Output: `[B, 2, H, W]` — two channels for horizontal and vertical displacement. Applications: video compression, action recognition, video stabilization, autonomous driving.

> **📊 Planned diagram:** Optical flow visualization — arrow field overlaid on a video frame, color-wheel encoding of magnitude and direction.

## Classical Methods

> **Planned content:** Brightness constancy assumption. Horn-Schunck: global regularization (smooth flow field). Lucas-Kanade: local patch-based solution, pyramidal extension for large motions. FlowFields, EpicFlow. Limitations: occlusion, large displacements, texture-less regions.

> **📊 Planned diagram:** Lucas-Kanade optical flow — patch around a pixel, gradient matrix, iterative refinement.

## Deep Learning Approaches

> **Planned content:** FlowNet: first CNN for optical flow — correlation layer for matching. FlowNet2. SPyNet: spatial pyramid + warping. PWC-Net: pyramid + warping + cost volume. RAFT: recurrent all-pairs field transforms — the current state-of-the-art approach.

> **📊 Planned diagram:** RAFT architecture — feature encoder, context encoder, all-pairs correlation volume `[B, H*W, H*W]`, GRU update operator for iterative refinement.

## Evaluation: Endpoint Error (EPE)

> **Planned content:** Average endpoint error (AEE/EPE) = mean Euclidean distance between predicted and ground truth flow vectors. Fl-all: fraction of pixels with EPE > 3px AND > 5% of GT magnitude. Sintel and KITTI benchmarks.

## Applications

> **Planned content:** Video stabilization (inverse flow warping). Video super-resolution. Action recognition (two-stream networks with flow). Motion segmentation. Event cameras and flow.

**Next: [Chapter 11 — Video Understanding →](./11_video_understanding.md)**

---
*Last updated: May 2026*
