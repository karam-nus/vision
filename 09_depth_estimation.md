---
title: "Chapter 9 — Depth Estimation"
---

[← Back to Table of Contents](./README.md)

# Chapter 9 — Depth Estimation

> *"Depth perception bridges the gap between 2D images and the 3D world — the foundation for spatial understanding."*

## The Depth Estimation Problem

> **Planned content:** Output a dense depth map `[B, 1, H, W]` where each pixel value represents metric distance (meters) or relative disparity. Supervised (metric depth) vs. self-supervised (relative depth) approaches. Monocular vs. stereo vs. multi-view vs. sensor-guided (LiDAR).

> **📊 Planned diagram:** Depth estimation output — RGB image alongside depth map rendered as heatmap, with near (warm) and far (cool) color coding.

## Stereo Vision and Epipolar Geometry

> **Planned content:** Two cameras separated by baseline. Epipolar constraint: a 3D point lies on a known epipolar line in the other image. Disparity `d = f * B / Z` (f=focal length, B=baseline, Z=depth). Disparity maps from stereo matching. SGM (Semi-Global Matching). PSMNet, GwcNet.

> **📊 Planned diagram:** Stereo geometry — two cameras, epipolar line, disparity vs. depth relationship. Formula labeled diagram.

$$Z = \frac{f \cdot B}{d}, \quad d = x_L - x_R$$

## Monocular Depth Estimation

> **Planned content:** Ill-posed without additional cues. Supervised: use LiDAR ground truth during training. Self-supervised: pose network + photometric consistency loss (Monodepth2). Scale ambiguity in self-supervised methods. Foundation models for depth: DPT, Depth Anything, Marigold, UniDepth.

> **📊 Planned diagram:** Monocular depth cues — size, occlusion, atmospheric haze, texture gradient, linear perspective — how models learn to exploit them.

> **📊 Planned diagram:** Monodepth2 self-supervised training — source frame → depth net → depth + pose net → warp → photometric loss.

## Depth Completion

> **Planned content:** Sparse LiDAR points + RGB image → dense depth map. Sensor fusion problem. Guided upsampling. CSPN, DySPN. KITTI depth completion benchmark.

## LiDAR and Sensor Fusion

> **Planned content:** LiDAR gives sparse but highly accurate depth. Camera gives dense but unscaled color. Fusion: project LiDAR onto image plane, use as guide. Early fusion vs. late fusion vs. feature fusion. Surround-view setups (autonomous driving).

> **📊 Planned diagram:** LiDAR-camera projection — 3D point cloud overlaid on image, showing sparse guidance points.

## Evaluation Metrics

> **Planned content:** AbsRel (absolute relative error). SqRel (squared relative error). RMSE (root mean squared error). RMSE_log. Threshold metrics δ1, δ2, δ3 (fraction of pixels where max(d_pred/d_gt, d_gt/d_pred) < 1.25^n). SILog (scale-invariant log error). When each metric matters.

> **📊 Planned table:** Depth metrics summary — formula, interpretation, range, what failure modes each catches.

**Next: [Chapter 10 — Optical Flow →](./10_optical_flow.md)**

---
*Last updated: May 2026*
