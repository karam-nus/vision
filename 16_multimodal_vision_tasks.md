---
title: "Chapter 16 — Multimodal Vision Tasks"
---

[← Back to Table of Contents](./README.md)

# Chapter 16 — Multimodal Vision Tasks

> *"Vision alone is silent. Language alone is blind. Together, they describe and generate the visual world."*

## Visual Question Answering (VQA)

> **Planned content:** Input: image + natural language question. Output: answer (classification over answer vocabulary or open-ended generation). VQA v2 dataset. Bilinear fusion of visual and text features. Modular networks. Transformer-based VQA. Bias problems in VQA datasets. GQA compositional reasoning benchmark.

> **📊 Planned diagram:** VQA pipeline — image → visual encoder `[B, 196, 768]` + question → text encoder `[B, T, 768]` → cross-attention fusion → answer head.

## Image Captioning

> **Planned content:** Generate a natural language description of an image. Show-and-Tell: CNN encoder + LSTM decoder. Attention-based captioning (Show, Attend and Tell). Transformer-based (COCO captioning). BLEU, CIDEr, SPICE evaluation metrics. Dense captioning (describe regions).

> **📊 Planned diagram:** Image captioning with attention — at each decoding step, the attention map shows which image regions the model focuses on.

## Visual Grounding

> **Planned content:** Referring expression comprehension: given expression "the person in the red hat on the left", locate it. Phrase grounding: align noun phrases to bounding boxes. Visual grounding requires joint understanding of language and vision. TransVG, MDETR, Grounding DINO, Florence-2.

## Text-to-Image Generation

> **Planned content:** DALL-E, Stable Diffusion, Imagen, Midjourney. Diffusion model basics. CLIP text encoder as conditioning. CFG (classifier-free guidance). Evaluation: FID, IS, human preference. ControlNet: additional spatial conditioning (pose, depth, canny edge).

> **📊 Planned diagram:** Text-to-image pipeline — text prompt → CLIP text encoder → cross-attention conditioning → U-Net denoising → decoder → image.

## Vision-Language Pre-training

> **Planned content:** CLIP: contrastive learning on 400M image-text pairs. Image encoder (ViT) + text encoder (Transformer) + contrastive loss. Zero-shot transfer. ALIGN (1.8B pairs). CoCa: generative + contrastive. BLIP: filter noisy web data with captioner + filter. Why large-scale pre-training unlocks transfer.

> **📊 Planned diagram:** CLIP training — image batch `[B, C, H, W]` → image encoder → `[B, D]` + text batch → text encoder → `[B, D]` → cosine similarity matrix `[B, B]` → contrastive loss diagonal.

## Image-Text Retrieval

> **Planned content:** Given a text query, retrieve the matching image (or vice versa). Dual-encoder: embed both modalities, rank by similarity. Cross-encoder: jointly encode for better accuracy but slower. Recall@K metric. COCO retrieval, Flickr30K benchmarks.

## Video-Language Models

> **Planned content:** VideoCLIP, InternVideo, Video-LLaVA. Temporal understanding of video + language. Dense video captioning. Video QA (ActivityNet-QA, MSVD-QA). Challenges: temporal alignment, long video understanding.

**Next: [Chapter 17 — CNN Fundamentals →](./17_cnn_fundamentals.md)**

---
*Last updated: May 2026*
