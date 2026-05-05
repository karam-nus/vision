---
title: "Chapter 0 — Grand Overview of Computer Vision"
---

[← Back to Table of Contents](./README.md)

# Chapter 0 — Grand Overview of Computer Vision

> *"Seeing is not believing — it is a complex inference problem solved by billions of neurons working in parallel."*

## What Is Computer Vision?

> **Planned content:** Definition of CV as a field. The core inversion problem: from pixel arrays to structured semantic understanding. Why it is hard (viewpoint variation, illumination, occlusion, deformation, intra-class variation, background clutter). The duality between perception and generation.

> **📊 Planned diagram:** Illustration of the "inverse graphics" framing — CV as the inverse of rendering.

## The Anatomy of a Computer Vision System

> **Planned content:** Every modern CV system decomposes into the same pipeline regardless of task. Sensor → preprocessing → feature extraction → task head → post-processing → output.

> **📊 Planned diagram (flowchart):** Full end-to-end CV system pipeline — camera/sensor → frame capture → resize/normalize → backbone → neck/FPN → task head → decode/NMS/threshold → output annotations.

> **📊 Planned diagram:** Tensor shapes at each stage of the pipeline, from raw image `[H, W, 3]` (uint8) to final prediction.

## The Computer Vision Task Taxonomy

> **Planned content:** Comprehensive taxonomy of CV tasks. How they differ in output representation, label granularity, and difficulty. Which tasks are prerequisites for others.

> **📊 Planned diagram (tree):** Task taxonomy tree — input understanding (classification, detection, segmentation, depth, flow, pose, OCR, face) vs. generation (super-resolution, inpainting, text-to-image, video generation) vs. 3D (NeRF, reconstruction, SLAM).

### Image-Level Tasks
> **Planned content:** Image classification (single label, multi-label, fine-grained), image retrieval, scene recognition, image quality assessment, aesthetic scoring.

### Region / Instance-Level Tasks
> **Planned content:** Object detection (2D, 3D), instance segmentation, panoptic segmentation, salient object detection, referring expression comprehension.

### Pixel-Level Tasks
> **Planned content:** Semantic segmentation, depth estimation, optical flow, surface normal estimation, intrinsic image decomposition.

### Keypoint / Structure Tasks
> **Planned content:** Human pose estimation (2D, 3D, whole-body), hand pose, face alignment, skeleton-based action recognition.

### Identity and Biometric Tasks
> **Planned content:** Face detection, face recognition/verification, person re-identification, gait recognition, iris recognition.

### Video / Temporal Tasks
> **Planned content:** Video classification, temporal action localization, action detection, multi-object tracking, video object segmentation, video prediction.

### 3D and Scene-Level Tasks
> **Planned content:** Monocular/stereo depth, 3D object detection, point cloud processing, NeRF/3DGS, SLAM, multi-view stereo.

### Multimodal Tasks
> **Planned content:** VQA, image captioning, image-text retrieval, visual grounding, text-to-image generation, document understanding.

## A Brief History of Computer Vision

> **Planned content:** From Hubel & Wiesel's edge detectors (1959) through hand-crafted features (SIFT, HOG), shallow learning (SVM on features), deep learning revolution (AlexNet 2012), detection era (RCNN → YOLO), segmentation era (FCN → Mask R-CNN), attention revolution (ViT 2020), foundation model era (CLIP, SAM, DINOv2).

> **📊 Planned diagram (timeline):** Key milestones in CV history — illustrated timeline with key papers and accuracy jumps on ImageNet/COCO.

## The Model Performance Landscape

> **Planned content:** ImageNet top-1 accuracy vs. year. COCO mAP evolution. The accuracy-efficiency Pareto frontier. How to read a benchmark table.

> **📊 Planned diagram:** Scatter plot of ImageNet top-1 accuracy vs. GFLOPs for key architectures — AlexNet, VGG, ResNet, EfficientNet, ViT, Swin, ConvNeXt.

> **📊 Planned diagram:** COCO detection mAP vs. inference speed for key detectors.

## Compute Requirements

> **Planned content:** Training compute (GPU-hours) for landmark models. Inference compute (FLOPs, latency). Memory requirements (parameters × precision). The gap between research compute and edge-deployable compute.

> **📊 Planned table:** Key models — parameters, FLOPs, training time, typical inference latency (GPU/CPU/mobile).

## How This Guide Is Organized

> **Planned content:** Explanation of the six dimensions (use cases, architectures, optimization, data & training, deployment, frontier). How chapters relate to each other. Suggested learning paths for different roles.

> **📊 Planned diagram:** Chapter dependency graph — which chapters build on which.

**Next: [Chapter 1 — Image Fundamentals →](./01_image_fundamentals.md)**

---
*Last updated: May 2026*
