---
title: "Computer Vision — The Complete Guide"
permalink: /
---

# 👁️ Computer Vision — The Complete Guide

> **From Pixels to Predictions**: Understanding, building, and deploying ML-based computer vision systems. A comprehensive reference for practitioners, engineers, and researchers navigating the modern CV landscape.

## Who This Is For

You understand Python and have touched PyTorch or TensorFlow. You've used a pre-trained model, maybe fine-tuned one. But you want to truly understand **what is happening inside**: how a convolutional filter fires, why NMS is tricky, how ArcFace creates separable embedding spaces, how YOLO's anchor-free head works, and what makes a transformer see an image differently than a CNN. This guide takes you from first principles — images as tensors — all the way to foundation models, generative systems, and the research frontier.

## 📋 Table of Contents

| # | Chapter | What You'll Learn |
|---|---------|-------------------|
| **Overview & Foundations** | | |
| 0 | [Grand Overview](./00_grand_overview.md) | The full CV landscape — tasks, history, benchmarks, and how every piece connects |
| 1 | [Image Fundamentals](./01_image_fundamentals.md) | Images as tensors `[B, C, H, W]`, color spaces, preprocessing pipelines, camera models |
| **Use Cases** | | |
| 2 | [Image Classification](./02_image_classification.md) | Softmax, cross-entropy, fine-grained recognition, zero-shot classification, ImageNet |
| 3 | [Object Detection](./03_object_detection.md) | Two-stage vs one-stage, anchors, **NMS deep dive**, IoU variants, mAP |
| 4 | [Semantic Segmentation](./04_semantic_segmentation.md) | Encoder-decoder, dilated convolutions, ASPP, loss functions, mIoU |
| 5 | [Instance Segmentation](./05_instance_segmentation.md) | Mask R-CNN, panoptic segmentation, SAM, query-based methods |
| 6 | [Object Tracking](./06_object_tracking.md) | Kalman filter, **Hungarian algorithm deep dive**, Re-ID, SORT → ByteTrack |
| 7 | [Pose Estimation](./07_pose_estimation.md) | Keypoints, heatmaps, **PAF deep dive**, top-down vs bottom-up, 3D pose |
| 8 | [Face Recognition](./08_face_recognition.md) | Detection, alignment, embedding, **ArcFace/CosFace/SphereFace deep dive** |
| 9 | [Depth Estimation](./09_depth_estimation.md) | Stereo, monocular depth, LiDAR fusion, depth completion |
| 10 | [Optical Flow](./10_optical_flow.md) | Lucas-Kanade, FlowNet, RAFT, video stabilization |
| 11 | [Video Understanding](./11_video_understanding.md) | Action recognition, temporal models (3D conv, transformers), action localization |
| 12 | [3D Vision](./12_3d_vision.md) | Point clouds, NeRF, 3D Gaussian splatting, 3D object detection, SLAM |
| 13 | [Anomaly Detection](./13_anomaly_detection.md) | One-class learning, PatchCore, flow-based, reconstruction-based |
| 14 | [OCR & Document Understanding](./14_ocr_and_document_understanding.md) | Text detection, CRNN+CTC, end-to-end OCR, LayoutLM |
| 15 | [Medical Imaging](./15_medical_imaging.md) | Segmentation, classification, domain challenges, foundation models (MedSAM) |
| 16 | [Multimodal Vision Tasks](./16_multimodal_vision_tasks.md) | VQA, image captioning, visual grounding, text-to-image generation |
| **Architectures** | | |
| 17 | [CNN Fundamentals](./17_cnn_fundamentals.md) | Convolution math, receptive fields, normalization, depthwise & dilated convs |
| 18 | [Classification Architectures](./18_classification_architectures.md) | ResNet, MobileNet, EfficientNet, ViT, Swin, ConvNeXt — with param/FLOP tables |
| 19 | [Detection Architectures](./19_detection_architectures.md) | YOLO (v1→v11), Faster R-CNN, DETR, RT-DETR, FCOS, CenterNet |
| 20 | [Segmentation Architectures](./20_segmentation_architectures.md) | FCN, U-Net, DeepLab v3+, Mask R-CNN, Mask2Former, SAM |
| 21 | [Pose Architectures](./21_pose_architectures.md) | Stacked Hourglass, HRNet, ViTPose, OpenPose, DEKR, WholeBody |
| 22 | [Face Architectures](./22_face_architectures.md) | MTCNN, RetinaFace, MobileFaceNet, ArcFace, AdaFace |
| 23 | [Tracking Architectures](./23_tracking_architectures.md) | SORT, DeepSORT, FairMOT, ByteTrack, StrongSORT, TrackFormer |
| 24 | [Vision Transformers](./24_vision_transformers.md) | ViT, DeiT, Swin, MAE, DINO, DINOv2 — attention in 2D |
| 25 | [Multimodal Architectures](./25_multimodal_architectures.md) | CLIP, BLIP-2, LLaVA, Grounding DINO, SAM 2, Flamingo |
| **Optimization** | | |
| 26 | [Optimization Overview](./26_optimization_overview.md) | The accuracy-speed-size triangle, profiling, optimization taxonomy |
| 27 | [Quantization](./27_quantization.md) | PTQ, QAT, INT8/FP8, mixed precision, per-channel vs per-tensor |
| 28 | [Pruning & Sparsification](./28_pruning_and_sparsification.md) | Structured/unstructured pruning, lottery ticket, movement pruning, ASP |
| 29 | [Knowledge Distillation](./29_knowledge_distillation.md) | Feature mimicking, attention transfer, task-specific KD, self-distillation |
| 30 | [Neural Architecture Search](./30_neural_architecture_search.md) | RL/evolutionary/gradient-based NAS, DARTS, Once-for-All, EfficientNet |
| 31 | [Efficient Inference](./31_efficient_inference.md) | Graph fusion, ONNX, TensorRT, OpenVINO, TFLite, CoreML, NCNN |
| **Data & Training** | | |
| 32 | [Datasets](./32_datasets.md) | ImageNet, COCO, ADE20K, LFW, Kinetics, KITTI, MVTec — every major benchmark |
| 33 | [Data Augmentation](./33_data_augmentation.md) | MixUp, CutMix, Mosaic, AutoAugment, Copy-Paste, TTA |
| 34 | [Training Recipes](./34_training_recipes.md) | Loss functions, AdamW/LARS/SAM optimizers, LR schedules, EMA, bag-of-tricks |
| 35 | [Self-Supervised Learning](./35_self_supervised_learning.md) | SimCLR, MoCo, BYOL, MAE, DINO — learning without labels |
| **Deployment & Ecosystem** | | |
| 36 | [Deployment](./36_deployment.md) | ONNX export, serving (Triton/TorchServe), edge (Jetson/mobile), batching |
| 37 | [Ecosystem & Libraries](./37_ecosystem_and_libraries.md) | timm, MMDetection, Detectron2, Ultralytics, OpenCV, Hugging Face, Roboflow |
| 38 | [Evaluation & Benchmarks](./38_evaluation_and_benchmarks.md) | mAP computation, OKS, HOTA, FID, Rank-1, FLOP counting |
| **Frontier** | | |
| 39 | [Vision–Language Models](./39_vision_language_models.md) | CLIP ecosystem, GPT-4V, Gemini Vision, VQA, grounding, generation |
| 40 | [Foundation Models](./40_foundation_models.md) | SAM, DINOv2, Florence-2, InternVL — universal visual features |
| 41 | [Generative Vision](./41_generative_vision.md) | GANs, VAEs, DDPM, Stable Diffusion, ControlNet, video generation |
| 42 | [The Frontier](./42_frontier.md) | World models, 3DGS, embodied AI, neuromorphic vision, open problems |
| **Appendices** | | |
| A | [History of Computer Vision](./appendix_a_history.md) | Full timeline from Hubel & Wiesel to SAM 2 — every milestone and paradigm shift |
| B | [Math Foundations](./appendix_b_math_foundations.md) | Linear algebra, convolutions as matrix ops, Fourier, probability for CV practitioners |
| C | [Glossary](./appendix_c_glossary.md) | Every key term, acronym, and metric defined in one place |

## 🗺️ Learning Paths

<div class="diagram">
<div class="diagram-title">Recommended Learning Path</div>
<div class="flow">
  <div class="flow-node accent wide">👁️ Ch 0–1: Grand Overview & Image Fundamentals</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">🎯 Ch 2–16: Use Cases — The Full Task Taxonomy</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node purple wide">🏗️ Ch 17–25: Architectures — CNN to Multimodal</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node orange wide">⚡ Ch 26–31: Optimization — Speed, Size, Efficiency</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node cyan wide">📦 Ch 32–35: Data, Augmentation & Self-Supervised</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node teal wide">🚀 Ch 36–38: Deployment, Ecosystem & Evaluation</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node pink wide">🔬 Ch 39–42: Frontier — VLMs, Foundation Models, Generation</div>
</div>
</div>

### Path A: "I want to build a CV application" (8 chapters)
1. [00 — Grand Overview](./00_grand_overview.md) — the big picture
2. [01 — Image Fundamentals](./01_image_fundamentals.md) — images as tensors
3. [02 — Image Classification](./02_image_classification.md) — the simplest task
4. [03 — Object Detection](./03_object_detection.md) — the most deployed task
5. [18 — Classification Architectures](./18_classification_architectures.md) — choosing your backbone
6. [19 — Detection Architectures](./19_detection_architectures.md) — YOLO and friends
7. [33 — Data Augmentation](./33_data_augmentation.md) — training data strategy
8. [36 — Deployment](./36_deployment.md) — getting to production

### Path B: "I want to go deep on a specific task" (task chapter → architecture chapter → eval chapter)
- **Detection**: [03](./03_object_detection.md) → [19](./19_detection_architectures.md) → [38](./38_evaluation_and_benchmarks.md)
- **Segmentation**: [04](./04_semantic_segmentation.md) + [05](./05_instance_segmentation.md) → [20](./20_segmentation_architectures.md) → [38](./38_evaluation_and_benchmarks.md)
- **Face Recognition**: [08](./08_face_recognition.md) → [22](./22_face_architectures.md) → [38](./38_evaluation_and_benchmarks.md)
- **Tracking**: [06](./06_object_tracking.md) → [23](./23_tracking_architectures.md) → [38](./38_evaluation_and_benchmarks.md)
- **Pose Estimation**: [07](./07_pose_estimation.md) → [21](./21_pose_architectures.md) → [38](./38_evaluation_and_benchmarks.md)

### Path C: "I want to optimize for edge/mobile" (7 chapters)
1. [17 — CNN Fundamentals](./17_cnn_fundamentals.md) — understand what you're optimizing
2. [18 — Classification Architectures](./18_classification_architectures.md) — MobileNet, EfficientNet
3. [26 — Optimization Overview](./26_optimization_overview.md) — the landscape
4. [27 — Quantization](./27_quantization.md) — INT8 and beyond
5. [28 — Pruning](./28_pruning_and_sparsification.md) — structured pruning
6. [29 — Knowledge Distillation](./29_knowledge_distillation.md) — compress with teachers
7. [31 — Efficient Inference](./31_efficient_inference.md) — TensorRT, TFLite, CoreML

### Path D: "I want to understand vision transformers and foundation models" (6 chapters)
1. [24 — Vision Transformers](./24_vision_transformers.md) — ViT to DINOv2
2. [25 — Multimodal Architectures](./25_multimodal_architectures.md) — CLIP, BLIP-2, LLaVA
3. [35 — Self-Supervised Learning](./35_self_supervised_learning.md) — MAE, DINO
4. [39 — Vision–Language Models](./39_vision_language_models.md) — CLIP ecosystem
5. [40 — Foundation Models](./40_foundation_models.md) — SAM, DINOv2
6. [41 — Generative Vision](./41_generative_vision.md) — diffusion models

### Path E: "I want the research frontier" (full guide)
Read chapters 0 through 42 in order. Appendices A, B, and C provide depth on history, math, and terminology.

## 📚 Prerequisites

- **Python** — NumPy, basic PyTorch tensor operations
- **Linear algebra** — matrix multiplication, eigenvectors, SVD
- **ML basics** — loss functions, gradient descent, backpropagation, CNNs
- **Probability** — softmax, cross-entropy, Gaussian distributions
- **Command line** — terminal, pip/conda, CUDA setup

## 🧭 How This Guide Is Organized

This guide spans **six dimensions** of computer vision:

<div class="diagram">
<div class="diagram-title">Six Dimensions of Computer Vision</div>
<div class="diagram-grid cols-3">
  <div class="diagram-card accent">
    <div class="card-icon">🎯</div>
    <div class="card-title">Use Cases</div>
    <div class="card-desc">Classification, detection, segmentation, tracking, pose, face, depth, flow, video, 3D, anomaly, OCR, medical, multimodal</div>
  </div>
  <div class="diagram-card green">
    <div class="card-icon">🏗️</div>
    <div class="card-title">Architectures</div>
    <div class="card-desc">CNNs, ResNets, MobileNets, EfficientNets, ViT, Swin, YOLO, DETR, HRNet, MobileFaceNet, CLIP, BLIP-2</div>
  </div>
  <div class="diagram-card purple">
    <div class="card-icon">⚡</div>
    <div class="card-title">Optimization</div>
    <div class="card-desc">Quantization (QAT/PTQ), pruning, sparsification, knowledge distillation, NAS, efficient inference runtimes</div>
  </div>
  <div class="diagram-card orange">
    <div class="card-icon">📦</div>
    <div class="card-title">Data & Training</div>
    <div class="card-desc">Benchmark datasets, augmentation strategies, loss functions, optimizers, self-supervised pre-training</div>
  </div>
  <div class="diagram-card cyan">
    <div class="card-icon">🚀</div>
    <div class="card-title">Deployment</div>
    <div class="card-desc">ONNX, TensorRT, OpenVINO, TFLite, Triton, edge (Jetson / mobile), serving patterns</div>
  </div>
  <div class="diagram-card pink">
    <div class="card-icon">🔬</div>
    <div class="card-title">Frontier</div>
    <div class="card-desc">VLMs, foundation models, generative vision, 3D scene representations, embodied AI, open problems</div>
  </div>
</div>
</div>

## 📝 Changelog

| Date | Changes |
|------|---------|
| May 2026 | Initial structure — 43 chapters + 3 appendices |

---

*Last updated: May 2026*
