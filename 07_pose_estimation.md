---
title: "Chapter 7 — Pose Estimation"
---

[← Back to Table of Contents](./README.md)

# Chapter 7 — Pose Estimation

> *"Pose estimation gives machines a skeletal understanding of the human body — the first step toward understanding action, intention, and motion."*

## The Pose Estimation Problem

> **Planned content:** Predict the 2D (or 3D) locations of anatomical keypoints (joints) on a human body. Standard COCO body: 17 keypoints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles). Output: `[B, K, 2]` for 2D (x, y) or `[B, K, 3]` for 3D (x, y, z). Confidence scores per keypoint. Skeleton definition as a set of (keypoint_i, keypoint_j) edges.

> **📊 Planned diagram:** Human skeleton with 17 COCO keypoints labeled — connecting lines forming the skeleton graph.

## Top-Down vs. Bottom-Up Approaches

> **Planned content:** Top-down: first detect person bounding boxes → crop → predict keypoints per person. Accurate, scales poorly with crowd density. Bottom-up: predict all keypoints in image at once → group into instances. Faster, handles crowds. Which approach to use when.

> **📊 Planned diagram:** Top-down vs. bottom-up pipeline comparison — with tensor shapes, trade-offs table, and example on crowded scene.

## Heatmap-Based Prediction

> **Planned content:** Ground truth representation: Gaussian blobs centered at each keypoint. Model outputs a heatmap `[B, K, H/4, W/4]` — one channel per keypoint. Decoding: argmax of each heatmap channel → (x, y) coordinates. Sub-pixel refinement using Taylor expansion or Gaussian fitting. Why heatmaps outperform direct coordinate regression.

> **📊 Planned diagram:** Heatmap generation — keypoint (x, y) → Gaussian blob on feature map, with σ (standard deviation) controlling spread.

```python
def generate_target(joints, joints_vis, heatmap_size, sigma=2):
    """
    joints:     [K, 2]   — ground truth keypoint coordinates
    joints_vis: [K]      — visibility flags (0=absent, 1=occluded, 2=visible)
    heatmap_size: (W, H) — output heatmap dimensions
    Returns: target [K, H, W], target_weight [K, 1]
    """
    num_joints = joints.shape[0]
    H, W = heatmap_size[1], heatmap_size[0]
    target = np.zeros((num_joints, H, W), dtype=np.float32)  # [K, H, W]

    for joint_id, (joint, vis) in enumerate(zip(joints, joints_vis)):
        if vis > 0:
            mu_x, mu_y = joint[0], joint[1]
            # Create meshgrid and compute Gaussian
            x = np.arange(0, W, 1, np.float32)       # [W]
            y = np.arange(0, H, 1, np.float32)[:, None]  # [H, 1]
            target[joint_id] = np.exp(
                -((x - mu_x)**2 + (y - mu_y)**2) / (2 * sigma**2)
            )  # [H, W]
    return target  # [K, H, W]
```

## ⭐ Part Affinity Fields (PAF) — Deep Dive

> **Planned content:** The bottom-up grouping problem: given K*N heatmap peaks (K keypoints, N persons), which peaks belong to the same person? PAFs: encode the direction of each limb (edge in skeleton) as a 2D vector field `[B, 2*L, H, W]` (L limbs). A PAF for limb (j1, j2) has a unit vector at each pixel pointing from j1 to j2 if inside the limb. Line integral scoring: integrate PAF along the line between candidate keypoints to score the connection. Bipartite matching to assemble the skeleton.

> **📊 Planned diagram (flowchart):** PAF-based grouping — from heatmap peaks to assembled skeletons, step by step: peak detection → limb score computation → bipartite matching per limb → NMS + merging.

> **📊 Planned diagram:** PAF visualization — color wheel showing direction at each pixel, overlaid on the limb region. Example for "left forearm" connecting left elbow to left wrist.

$$S_{jk} = \int_{u=0}^{1} \mathbf{L}_{c}(\mathbf{p}(u)) \cdot \frac{\mathbf{d}_{j2} - \mathbf{d}_{j1}}{\|\mathbf{d}_{j2} - \mathbf{d}_{j1}\|} \, du$$

> **📊 Planned diagram:** Line integral computation — sampling points along the limb candidate, dotting with PAF vector, summing to get limb score.

## Object Keypoint Similarity (OKS)

> **Planned content:** The pose evaluation metric. Per-keypoint Gaussian kernel based on ground truth visibility and the object scale. OKS ∈ [0, 1]. mAP over OKS thresholds 0.5:0.95. Per-keypoint weights in COCO (ankle harder to annotate than nose). How OKS relates to IoU for detection.

> **📊 Planned diagram:** OKS computation — predicted vs. GT keypoints, Gaussian kernels, per-keypoint scores → OKS.

$$\text{OKS} = \frac{\sum_i \exp\left(-\frac{d_i^2}{2 s^2 \sigma_i^2}\right) \cdot \delta(v_i > 0)}{\sum_i \delta(v_i > 0)}$$

## 3D Pose Estimation

> **Planned content:** Lifting 2D predictions to 3D (video-based lifting). Direct 3D regression from images. Weak perspective assumption. Root-relative vs. absolute 3D coordinates. MPJPE (Mean Per-Joint Position Error) metric. Datasets: Human3.6M, MPI-INF-3DHP. Multi-view triangulation.

> **📊 Planned diagram:** 2D-to-3D lifting pipeline — 2D keypoints `[K, 2]` → temporal context → 3D keypoints `[K, 3]` relative to root joint (pelvis).

## Whole-Body Pose

> **Planned content:** Beyond 17 body keypoints: add face (68 landmarks), hands (21 keypoints each), feet. 133-keypoint whole-body (COCO WholeBody). Challenges: different scales and visibility patterns for body/face/hands. Models: ZoomNet, ExPose, OSX.

> **📊 Planned diagram:** Whole-body keypoint map — body + face + hands + feet labeled on a single figure.

## Multi-Person Pose in Video

> **Planned content:** AlphaPose Video, PoseTrack. Temporal consistency. Pose tracking vs. pose estimation + tracking. PoseTrack18/21 benchmarks.

**Next: [Chapter 8 — Face Recognition →](./08_face_recognition.md)**

---
*Last updated: May 2026*
