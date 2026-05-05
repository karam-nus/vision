---
title: "Chapter 12 — 3D Vision"
---

[← Back to Table of Contents](./README.md)

# Chapter 12 — 3D Vision

> *"The world is 3D. Computer vision that ignores this is perpetually incomplete."*

## 3D Representations

> **Planned content:** Comparison of 3D representations: point clouds (unstructured, `[N, 3+]`), voxel grids (`[X, Y, Z]`, memory-expensive), meshes (vertices + faces), implicit surfaces (occupancy networks, SDF), NeRF (neural radiance field), 3D Gaussian Splatting. When to use each.

> **📊 Planned diagram:** Six 3D representations visualized side-by-side for the same 3D object (Stanford bunny) — point cloud, voxel grid, mesh, SDF slice, NeRF renders, 3DGS render.

## Point Cloud Processing

> **Planned content:** LiDAR point clouds: 360° scan, `[N, 4]` (x, y, z, intensity). Irregular, unordered. PointNet: per-point MLP + global max pooling — permutation invariant. PointNet++: hierarchical local grouping (ball query). Set abstraction layers.

> **📊 Planned diagram:** PointNet architecture — input `[B, N, 3]` → per-point MLP → max pooling → global feature `[B, 1024]` → classification head.

## 3D Object Detection

> **Planned content:** VoxelNet: voxelize point cloud → 3D convolutions → RPN. PointPillars: vertical pillars instead of voxels, faster. CenterPoint: center-based detection from BEV feature map. SECOND: sparse 3D convolutions. Evaluation: 3D IoU, BEV IoU, KITTI, Waymo Open Dataset.

> **📊 Planned diagram:** PointPillars pipeline — scatter points into vertical pillars → PointNet per pillar → 2D BEV feature map → 2D detection head.

## Neural Radiance Fields (NeRF)

> **Planned content:** Represent a scene as an implicit function: MLP maps `(x, y, z, θ, φ) → (RGB, σ)`. Volume rendering: march rays, sample along ray, composite colors and densities. Training: minimize photometric loss against multi-view images. Limitations: slow training, slow rendering, requires many views. Improvements: Instant-NGP (hash encoding), TensoRF, Mip-NeRF.

> **📊 Planned diagram:** NeRF rendering pipeline — ray casting, sampling `N` points per ray `[N, 5]` → MLP → density + color → volume rendering integral → pixel color.

> **📊 Planned diagram:** Positional encoding — why raw coordinates fail and how sinusoidal encoding helps the MLP fit high-frequency details.

$$C(r) = \int_{t_n}^{t_f} T(t)\,\sigma(r(t))\,c(r(t),d)\,dt, \quad T(t) = \exp\!\left(-\int_{t_n}^{t}\sigma(r(s))ds\right)$$

## 3D Gaussian Splatting (3DGS)

> **Planned content:** Represent scene as a set of 3D Gaussians with position, covariance, opacity, and SH color coefficients. Tile-based rasterizer: sort Gaussians by depth, project to 2D, alpha composite. Real-time rendering. Training from multi-view images with SfM initialization. Memory vs. NeRF.

> **📊 Planned diagram:** 3DGS pipeline — SfM point cloud → 3D Gaussians initialization → differentiable rasterization → photometric loss → densification/pruning.

## SLAM (Simultaneous Localization and Mapping)

> **Planned content:** Estimate camera pose while building a map. Visual SLAM: camera only (ORB-SLAM). Visual-inertial SLAM. Dense vs. sparse maps. Neural SLAM (iMAP, NICE-SLAM, MonoGS). Frontend (odometry) + backend (loop closure, bundle adjustment).

> **📊 Planned diagram:** SLAM system architecture — camera stream → feature tracking → pose estimation → map update → loop closure → global optimization.

**Next: [Chapter 13 — Anomaly Detection →](./13_anomaly_detection.md)**

---
*Last updated: May 2026*
