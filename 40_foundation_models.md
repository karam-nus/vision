---
title: "Chapter 40 — Foundation Models for Vision"
---

[← Back to Table of Contents](./README.md)

# Chapter 40 — Foundation Models for Vision

> *"A foundation model learns from the world so that downstream tasks need only a gentle nudge — not a complete rebuilding."*

## What Makes a Foundation Model?

> **Planned content:** Large scale (data + parameters). General-purpose: useful for many tasks without task-specific training. Emergent capabilities. Transfer via prompting, linear probing, or fine-tuning. Examples in vision: SAM, DINOv2, Florence-2, InternVL. Contrast with specialist models (one task only).

> **�� Planned diagram:** Foundation model vs. specialist model — one large pre-trained model → many downstream tasks via adaptation, vs. many specialist models each trained from scratch.

## SAM: Segment Anything Model (2023)

> **Planned content:** Architecture (covered in Ch. 5, 20) — scale and impact here. SA-1B: 11M images, 1.1B masks — largest segmentation dataset. Training: data engine (automatic annotation using SAM itself). Zero-shot generalization across domains. SAM's limitations: no semantic understanding, no video (until SAM 2).

> **📊 Planned diagram:** SAM data engine — manual annotation → semi-automatic → fully automatic → SA-1B dataset.

## DINOv2: Universal Visual Features (2023)

> **Planned content:** Pre-training pipeline (covered in Ch. 35) — downstream performance here. Linear probing: 86.2% ImageNet top-1 (ViT-g). k-NN: 83.5% without fine-tuning. Segmentation: zero-shot semantic segmentation. Depth estimation: strong linear probe. Monocular depth: scale+shift MSE regression on frozen features. Universal feature extractor.

> **📊 Planned diagram:** DINOv2 zero-shot capabilities — same frozen model linear probed for classification, segmentation, depth, retrieval.

## Florence-2: Unified Representation for Vision (2024)

> **Planned content:** Microsoft's unified vision foundation model. Single model trained for 12+ tasks with a seq2seq formulation (task token + image → text output). Tasks: captioning, detection, grounding, segmentation, OCR. FlorenceVL-1B, Florence-2-Base, Florence-2-Large. FLOP-efficient.

> **📊 Planned diagram:** Florence-2 unified task formulation — same image encoder + different task prompts → different output heads.

## InternVL: Scaling Vision Encoders

> **Planned content:** InternViT-6B: 6B parameter vision encoder. InternVL-Chat: multimodal LLM. Dynamic high-resolution: tile images into sub-images for fine-grained understanding. Competitive with GPT-4V on vision benchmarks with open weights.

## Using Foundation Models as Backbones

> **Planned content:** Replace task-specific backbones with DINOv2/SAM/CLIP. Linear probing results. Task-specific fine-tuning. Adapter layers. LoRA fine-tuning of ViT. When foundation model features help (small data regime, novel domains) vs. when task-specific training still wins.

```python
import torch
from torchvision.models import ViT_H_14_Weights
import timm

# DINOv2 as backbone
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14")
model.eval()

x = torch.randn(1, 3, 224, 224)  # [1, 3, 224, 224]
with torch.no_grad():
    features = model(x)           # [1, 1536]  — ViT-G/14 global feature
    patch_features = model.get_intermediate_layers(x, n=1)[0]  # [1, 256, 1536]
```

**Next: [Chapter 41 — Generative Vision →](./41_generative_vision.md)**

---
*Last updated: May 2026*
