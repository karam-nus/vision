---
title: "Chapter 6 — Object Tracking"
---

[← Back to Table of Contents](./README.md)

# Chapter 6 — Object Tracking

> *"Detection is per frame. Tracking is across time — maintaining identity through motion, occlusion, and crowding."*

## The Tracking Problem

> **Planned content:** Multi-Object Tracking (MOT) problem formulation. Input: sequence of frames. Output: per-frame set of tracks, each with consistent ID, bounding box, and class. SOT (single object) vs. MOT (multiple objects). Online vs. offline tracking. Detection-based tracking (tracking-by-detection) vs. joint detection+tracking.

> **📊 Planned diagram:** MOT problem — sequence of frames with bounding boxes, showing track continuity through occlusion and re-identification.

## The Tracking-by-Detection Paradigm

> **Planned content:** Step 1: detect objects in every frame. Step 2: associate detections across frames. The two-step pipeline: detector → association module. Strengths: modular, benefits from strong detectors. Weakness: detector failures propagate. SORT, DeepSORT, ByteTrack all follow this paradigm.

> **📊 Planned diagram (flowchart):** Tracking-by-detection pipeline — frame t → detector → detections → association with tracks from t-1 → updated tracks → output.

## Kalman Filter for Motion Prediction

> **Planned content:** State vector `[cx, cy, s, r, vc_x, vc_y, vs]` (centroid, area scale, aspect ratio, velocities). Predict step: propagate state with constant velocity model. Update step: correct with new measurement. Covariance matrices (Q for process noise, R for measurement noise). When to predict vs. update. Why Kalman filter is fast and effective despite being linear.

> **📊 Planned diagram:** Kalman filter cycle — state prediction → measurement → update, showing uncertainty ellipses before and after measurement.

$$\hat{x}_{t|t-1} = F \hat{x}_{t-1|t-1}, \qquad P_{t|t-1} = F P_{t-1|t-1} F^T + Q$$
$$K_t = P_{t|t-1} H^T (H P_{t|t-1} H^T + R)^{-1}$$
$$\hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t (z_t - H \hat{x}_{t|t-1})$$

## ⭐ Hungarian Algorithm — Deep Dive

> **Planned content:** The association problem: given M tracks and N detections, find the optimal matching that minimizes total cost (e.g., 1 - IoU). This is the classic Linear Assignment Problem (LAP). Hungarian algorithm: find minimum-cost perfect matching in a bipartite graph. Complexity O(n³). How cost matrices are built. Threshold-based acceptance: only accept matches below a cost threshold. Unmatched detections → new tracks. Unmatched tracks → lost state.

> **📊 Planned diagram (flowchart):** Hungarian algorithm step-by-step — cost matrix construction → row/column reduction → find zero coverage → assignment extraction.

> **📊 Planned diagram:** Bipartite matching visualization — tracks (left nodes) matched to detections (right nodes) with cost-weighted edges. Before and after assignment.

> **📊 Planned diagram:** Cost matrix example with IoU values — showing how the matrix is built and what the optimal assignment looks like.

```python
from scipy.optimize import linear_sum_assignment
import numpy as np

def associate(tracks, detections, iou_threshold=0.3):
    """
    tracks:     List of track boxes [M, 4]
    detections: List of detection boxes [N, 4]
    returns: (matches, unmatched_tracks, unmatched_detections)
    """
    if len(tracks) == 0 or len(detections) == 0:
        return [], list(range(len(tracks))), list(range(len(detections)))

    iou_matrix = compute_iou_matrix(tracks, detections)  # [M, N]
    cost_matrix = 1 - iou_matrix                         # [M, N]

    # Hungarian algorithm: find optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)  # [K], [K]

    matches, unmatched_t, unmatched_d = [], [], []

    # Filter out weak matches
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_threshold:
            matches.append((r, c))
        else:
            unmatched_t.append(r)
            unmatched_d.append(c)

    unmatched_t += [i for i in range(len(tracks))     if i not in row_ind]
    unmatched_d += [j for j in range(len(detections)) if j not in col_ind]

    return matches, unmatched_t, unmatched_d
```

## Track Lifecycle

> **Planned content:** Track states: Tentative (newly created, not yet confirmed) → Active (confirmed, being matched) → Lost (missed for max_age frames) → Deleted. Minimum hits to confirm a track. Maximum age before deletion. Birth, confirmation, occlusion handling, re-identification.

> **📊 Planned diagram (state machine):** Track lifecycle state machine — Tentative → Active → Lost → Deleted, with transition conditions.

## Appearance Features and Re-ID

> **Planned content:** Pure motion (IoU) fails under occlusion and fast motion. Appearance embeddings: re-ID network extracts a feature vector per detection crop. Cosine similarity in embedding space for long-range matching. Gallery management: rolling average of appearance features per track. DeepSORT: adds appearance cost to Kalman IoU cost.

> **📊 Planned diagram:** Re-ID pipeline — detection crop `[1, 3, 128, 64]` → Re-ID backbone → embedding `[1, 512]` → cosine similarity with track gallery.

## SORT, DeepSORT, ByteTrack — A Comparison

> **Planned content:** SORT: Kalman + Hungarian on IoU only. Simple and fast. DeepSORT: add appearance feature matching. ByteTrack: match ALL detections (even low-confidence) — two-round matching prevents missed detections. StrongSORT: improved Re-ID + motion model. OC-SORT: observation-centric Kalman. BoTrack. Analysis of where each method improves.

> **📊 Planned table:** SORT family comparison — method, detector used, matching strategy, whether Re-ID is used, MOT17 HOTA score.

## Tracking Evaluation Metrics

> **Planned content:** MOTA (Multiple Object Tracking Accuracy) — penalizes FP, FN, ID switches. MOTP (precision). IDF1 (identity F1 score). HOTA (Higher Order Tracking Accuracy) — decomposes into detection and association quality. How HOTA resolves disagreements between MOTA and IDF1.

> **📊 Planned diagram:** HOTA decomposition — DetA vs. AssA, HOTA = geometric mean.

**Next: [Chapter 7 — Pose Estimation →](./07_pose_estimation.md)**

---
*Last updated: May 2026*
