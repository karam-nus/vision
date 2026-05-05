---
title: "Chapter 37 — Ecosystem and Libraries"
---

[← Back to Table of Contents](./README.md)

# Chapter 37 — Ecosystem and Libraries

> *"Standing on the shoulders of giants — knowing the ecosystem well is a superpower for fast development."*

## OpenCV

> **Planned content:** The foundational CV library (C++/Python). Image I/O, color space conversion, resizing, drawing. Classical algorithms: Harris corners, SIFT/SURF (patented), ORB, optical flow (Lucas-Kanade), HOG, Viola-Jones face detection. DNN module: run ONNX/Caffe/TF models. VideoCapture/VideoWriter. Performance: C++ core, GStreamer pipeline. When to use OpenCV vs. PyTorch.

```python
import cv2
import numpy as np

# Load, process, display
img = cv2.imread("image.jpg")             # [H, W, 3], BGR uint8
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # [H, W, 3], RGB

# Run ONNX model with OpenCV DNN
net = cv2.dnn.readNetFromONNX("model.onnx")
blob = cv2.dnn.blobFromImage(img, 1/255.0, (640, 640), swapRB=True)
# blob: [1, 3, 640, 640] float32 normalized
net.setInput(blob)
output = net.forward()  # inference
```

## timm: PyTorch Image Models

> **Planned content:** The de-facto model zoo for PyTorch CV. 700+ pre-trained models. Unified API: `timm.create_model("efficientnet_b0", pretrained=True, num_classes=10)`. Feature extraction: `model.forward_features(x)`. Mixed precision support. Custom heads. Used as backbone for almost all SOTA methods.

```python
import timm

# List available models matching a pattern
timm.list_models("*efficientnet*")[:5]

model = timm.create_model(
    "efficientnet_b0",
    pretrained=True,
    num_classes=0,       # global avg pool → feature embedding
    global_pool="avg"
)
x = torch.randn(4, 3, 224, 224)   # [B=4, C=3, H=224, W=224]
features = model(x)                # [B=4, 1280]  — EfficientNet-B0 embedding
```

## OpenMMLab: MMDetection, MMPose, MMSegmentation

> **Planned content:** OpenMMLab (CUHK + SenseTime + community): modular framework. MMDetection: 40+ detection algorithms (Faster RCNN, YOLO, DETR etc.). MMPose: pose estimation (HRNet, ViTPose). MMSegmentation: semantic segmentation (DeepLab, Segformer). MMTracking, MMOCR, MMAction2. Config-driven design. TIMM integration. Powerful for research, sometimes complex for production.

## Detectron2 (Meta AI)

> **Planned content:** Facebook's detection + segmentation research platform. Faster R-CNN, Mask R-CNN, Panoptic FPN, DensePose, PointRend. Register/config system. Custom data loading. Trainer class. Good for Mask R-CNN variants and research extensions.

## Ultralytics (YOLOv5, YOLOv8, YOLO11)

> **Planned content:** The most practical detection framework. Simple API: `model = YOLO("yolov8n.pt"); results = model("image.jpg")`. Training, validation, export in one line. Multi-task support: detection, segmentation, pose, classification, OBB. Export to 12+ formats. Actively maintained.

```python
from ultralytics import YOLO

# Load pre-trained model
model = YOLO("yolov8n.pt")   # nano: 3.2M params, 8.7 GFLOPs

# Inference
results = model("image.jpg", conf=0.25, iou=0.45)
for r in results:
    boxes  = r.boxes.xyxy    # [N, 4]  xyxy bounding boxes
    confs  = r.boxes.conf    # [N]     confidence scores
    clses  = r.boxes.cls     # [N]     class indices

# Train
model.train(data="coco.yaml", epochs=100, imgsz=640)

# Export
model.export(format="onnx", opset=17, simplify=True)
```

## Hugging Face Transformers for Vision

> **Planned content:** ViT, Swin, DeiT, SegFormer, DETR, SAM, CLIP, BLIP-2, LLaVA. Unified AutoModel API. Pipeline abstraction. Hub for model weights and datasets. transformers + datasets + accelerate + peft stack.

## Roboflow

> **Planned content:** Dataset management for CV. Annotation (LabelImg, CVAT integration). Dataset versioning. Augmentation pipeline. Export in 30+ formats (YOLO, COCO, Pascal VOC, etc.). Universe: 50K+ public CV datasets. Roboflow Inference for deployment.

## Weights & Biases and MLflow

> **Planned content:** W&B: experiment tracking, hyperparameter sweeps, model registry, dataset versioning, interactive visualizations. MLflow: open-source, model registry, tracking server. Both integrate with all major frameworks.

## Annotation Tools

> **Planned content:** LabelImg: simple bounding box annotation. CVAT (Intel): web-based, supports all annotation types (detection, segmentation, pose, tracking). Label Studio: multi-modal, flexible. Scale AI / Labelbox: commercial annotation platforms. Semi-automatic annotation with SAM.

**Next: [Chapter 38 — Evaluation & Benchmarks →](./38_evaluation_and_benchmarks.md)**

---
*Last updated: May 2026*
