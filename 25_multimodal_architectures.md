---
title: "Chapter 25 — Multimodal Architectures"
---

[← Back to Table of Contents](./README.md)

# Chapter 25 — Multimodal Architectures

> *"Multimodal architectures are the connective tissue between seeing and understanding — bridging the pixel and the word."*

## CLIP: Contrastive Language-Image Pre-training (2021)

> **Planned content:** Dual encoder: ViT image encoder + Transformer text encoder. Contrastive training on 400M (image, text) pairs. InfoNCE loss on the `[B, B]` cosine similarity matrix. Zero-shot classification: compare image embedding to text embeddings of class names. OpenCLIP: open reproduction. SigLIP: sigmoid loss instead of softmax for better scaling.

> **📊 Planned diagram:** CLIP architecture — image encoder `[B, D]` + text encoder `[B, D]` → cosine similarity matrix `[B, B]` → diagonal = positives, off-diagonal = negatives → InfoNCE loss.

> **📊 Planned diagram:** CLIP zero-shot classification — image embedding cosine similarity with N text template embeddings `["a photo of a {class}"]`.

```python
import torch
import clip

model, preprocess = clip.load("ViT-B/32")

image = preprocess(image_pil).unsqueeze(0)        # [1, 3, 224, 224]
text  = clip.tokenize(["a cat", "a dog", "a car"]) # [3, 77]

with torch.no_grad():
    image_features = model.encode_image(image)     # [1, 512]
    text_features  = model.encode_text(text)       # [3, 512]

    # Normalize and compute similarity
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features  = text_features  / text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)  # [1, 3]
```

## BLIP-2: Bootstrapped VLMs (2023)

> **Planned content:** Bridge frozen image encoder (ViT) and frozen LLM (OPT, Flan-T5) with a learnable Q-Former (Querying Transformer). Q-Former: N learnable queries `[B, 32, 768]` attend to image features via cross-attention. Two-stage training: image-text matching first, then image-to-text generation.

> **📊 Planned diagram:** BLIP-2 architecture — frozen ViT → Q-Former (32 learnable queries + cross-attention to image) → query tokens `[B, 32, 768]` → project → prefix tokens for LLM → text generation.

## LLaVA: Language-Vision Assistant (2023)

> **Planned content:** Simple approach: project ViT features to LLM token space with a linear (or MLP) projection. LLaVA-1: linear projection. LLaVA-1.5: MLP projection + higher resolution. LLaVA-NeXT (1.6): AnyRes for high-resolution image handling. Instruction fine-tuning on visual instruction data.

> **📊 Planned diagram:** LLaVA architecture — CLIP ViT `[B, 256, 1024]` → projection MLP → visual tokens `[B, 256, 4096]` → prepend to text tokens → LLM.

## Grounding DINO: Open-Set Object Detection (2023)

> **Planned content:** Extend DINO (detection transformer) with language conditioning. Text encoder (BERT) + image encoder + fusion transformer → detect any class described in text. Open-vocabulary detection: zero-shot detection of any noun. Combination with SAM for "grounded segment anything".

> **📊 Planned diagram:** Grounding DINO — image + text → dual encoder → feature fusion → decoder → grounded boxes for text-described objects.

## SAM 2: Segment Anything in Images and Video (2024)

> **Planned content:** Streaming memory encoder for temporal consistency in video. Hiera image encoder. Memory bank: object pointers from past frames. Memory attention: cross-attend to past frames. Real-time interactive video segmentation. SA-V dataset: 51K videos.

> **📊 Planned diagram:** SAM 2 memory architecture — current frame encoder + memory from t-1, t-2 → memory cross-attention → mask decoder.

## Flamingo and Idefics (Few-Shot VLMs)

> **Planned content:** Flamingo: interleaved image-text in context. Perceiver Resampler compresses image features. Cross-attention layers inserted into frozen LLM. Few-shot VQA via in-context examples. Idefics: open reproduction of Flamingo.

## InternVL: International Large Vision Model

> **Planned content:** Scaling ViT to ViT-6B. Joint training with CLIP-style and generative objectives. InternVL-Chat: multimodal conversation. Strong on multilingual benchmarks.

**Next: [Chapter 26 — Optimization Overview →](./26_optimization_overview.md)**

---
*Last updated: May 2026*
