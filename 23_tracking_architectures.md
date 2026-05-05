---
title: "Chapter 23 — Tracking Architectures"
---

[← Back to Table of Contents](./README.md)

# Chapter 23 — Tracking Architectures

> *"Tracking architectures encode assumptions about motion, appearance, and identity — choosing the wrong one costs IDs."*

## SORT: Simple Online and Realtime Tracking (2016)

> **Planned content:** Kalman filter for motion prediction + Hungarian algorithm for IoU-based assignment. No appearance features. 7-line core algorithm. Surprisingly competitive on pedestrian tracking. Limitations: ID switches under occlusion. FPS: 260 Hz (detector not included).

> **📊 Planned diagram:** SORT flow — predict Kalman states → detect → compute IoU matrix → Hungarian assignment → update states → manage track lifecycle.

## DeepSORT (2017)

> **Planned content:** Add appearance descriptor (Re-ID network) to SORT. Two-distance metric: IoU distance + cosine distance on appearance embeddings. Cascade matching: prefer recently-seen tracks. Track state: confirmed vs. tentative. 20Hz with deep features.

> **📊 Planned diagram:** DeepSORT metric fusion — IoU distance matrix `[M, N]` + appearance cosine distance `[M, N]` → element-wise gating → cascade assignment.

## FairMOT (2020)

> **Planned content:** Joint detection and re-ID in a single network. CenterNet detector head + re-ID embedding head. One-shot tracker. Feature at detected center = re-ID embedding. Fairer competition between detection and re-ID gradients.

> **📊 Planned diagram:** FairMOT head — shared DLA backbone → detection branch (heatmap + offset + size) + re-ID branch (128D embedding per center).

## ByteTrack: Every Detection Counts (2022)

> **Planned content:** Key insight: low-confidence detections (0.1-0.5) still carry useful identity information — use them in a second matching round. Two-round matching: high-confidence detections first → unmatch tracks → low-confidence detections second. Dramatically reduces missed detections (FN → ID switches).

> **📊 Planned diagram (flowchart):** ByteTrack two-round matching — high-conf detections + all tracks → Hungarian round 1 → remaining tracks + low-conf detections → Hungarian round 2 → new tracks from unmatched high-conf.

## StrongSORT (2022)

> **Planned content:** Improved re-ID model (OSNet) + EMA (exponential moving average) for appearance features + motion compensation (camera motion). AFLink: inter-frame non-linear link prediction to fill gaps. GSSI: Gaussian-smoothed state interpolation.

## OC-SORT: Observation-Centric SORT (2022)

> **Planned content:** Problem with standard Kalman: when a track is lost (occluded), virtual observations cause drift in state estimation. OC-SORT: re-initialize Kalman with actual observations when track is recovered. Virtual observations only for motion compensation.

> **📊 Planned diagram:** OC-SORT observation-centric re-correction — showing Kalman state drift during occlusion vs. observation-anchored correction.

## Transformer-Based Trackers

> **Planned content:** TrackFormer: extend DETR with track queries (propagated object queries from t-1). MOTR: end-to-end multi-object tracking with motion-aware transformer. Advantages: no explicit IoU/appearance matching, learned association. Challenges: slower, hard to debug.

> **📊 Planned diagram:** TrackFormer — DETR object queries for new detections + track queries (re-used from t-1) → joint update → maintained object identity.

## Metrics Deep Dive

> **Planned content:** MOTA = 1 - (FN + FP + IDSW) / GT. IDF1 = 2 * IDTP / (2*IDTP + IDFP + IDFN). HOTA = √(DetA × AssA). Why HOTA is more balanced. MOT17, MOT20, DanceTrack, SportsMOT benchmarks.

> **📊 Planned diagram:** HOTA decomposition visualization — DetA vs AssA space, showing where different trackers fall.

**Next: [Chapter 24 — Vision Transformers →](./24_vision_transformers.md)**

---
*Last updated: May 2026*
