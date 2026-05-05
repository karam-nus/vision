---
title: "Chapter 39 — Vision-Language Models"
---

[← Back to Table of Contents](./README.md)

# Chapter 39 — Vision–Language Models

> *"Vision-language models don't just see — they understand, describe, answer, and generate. They are the bridge between pixels and knowledge."*

## The Vision-Language Pre-training Revolution

> **Planned content:** History: from dual-encoder retrieval (VSE++) to contrastive pre-training at scale (CLIP). The data flywheel: internet image-text pairs at web scale. Zero-shot transfer: no task-specific fine-tuning needed. The emergent behaviors from scale.

> **📊 Planned diagram:** VLM evolution timeline — VSE++ → ImageBERT → CLIP → ALIGN → BLIP → BLIP-2 → LLaVA → GPT-4V → Gemini Vision.

## CLIP Ecosystem

> **Planned content:** Original CLIP (OpenAI). OpenCLIP: open reproduction, ViT-L/14, ViT-G/14. SigLIP: sigmoid loss, better scaling. EVA-CLIP: CLIP + MAE for ViT-18B. MetaCLIP: curated training data. CLIPA: larger crops, better efficiency. Applications: zero-shot classification, image-text retrieval, image search, visual RLHF reward.

> **📊 Planned diagram:** CLIP zero-shot pipeline — image features vs. class name text features, similarity-based classification without any fine-tuning.

## Open-Vocabulary Detection and Segmentation

> **Planned content:** OWL-ViT: CLIP ViT backbone + detection head, text queries. Grounding DINO: open-set detection with language grounding. GLIP: phrase grounding. GDINO-SAM: ground + segment pipeline. APE, GenerateU. These models enable detection of arbitrary object categories.

> **📊 Planned diagram:** Open-vocabulary detection pipeline — text query "person carrying a bag" → Grounding DINO → bounding box.

## GPT-4V, Gemini, and Claude Vision

> **Planned content:** GPT-4V: multimodal GPT-4, image + text interleaved. Capabilities: OCR, chart reading, spatial reasoning, diagram understanding. Gemini 1.5 Pro: native multimodal, 1M context, video understanding. Claude-3: vision capabilities. Limitations: hallucination, counting errors, fine-grained spatial reasoning.

> **📊 Planned diagram:** Capability comparison across closed-source VLMs — VQA, chart QA, OCR, code from image, video understanding.

## Video-Language Models

> **Planned content:** VideoCLIP, InternVideo2, Video-LLaVA. Challenges: temporal alignment, long video. Temporal position encoding. Frame sampling strategies. Video QA benchmarks (ActivityNet-QA, NExT-QA, MSVD-QA). EgoSchema (egocentric video understanding).

## VLMs for Dense Prediction

> **Planned content:** SegCLIP: CLIP for semantic segmentation. CLIPSeg: image + text + visual prompts → segmentation. LSeg: language-guided segmentation. NACLIP: no training zero-shot segmentation. SAM + CLIP pipeline for open-world segmentation.

## Training a Custom VLM

> **Planned content:** Data collection: image-text pairs (LAION, CC3M, CC12M). Data quality: quality filtering with CLIP score. Training infrastructure: contrastive loss, gradient checkpointing, ZeRO. Fine-tuning for downstream tasks (linear probe, zero-shot, full fine-tune). LLaVA instruction tuning recipe.

**Next: [Chapter 40 — Foundation Models →](./40_foundation_models.md)**

---
*Last updated: May 2026*
