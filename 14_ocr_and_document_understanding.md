---
title: "Chapter 14 — OCR and Document Understanding"
---

[← Back to Table of Contents](./README.md)

# Chapter 14 — OCR and Document Understanding

> *"OCR transforms images of language into machine-readable text — bridging vision and NLP."*

## Text in Images: Problem Space

> **Planned content:** Scene text (arbitrary orientation, perspective, fonts) vs. document text (structured, constrained). Text detection → text recognition → information extraction pipeline. Challenges: curved text, low resolution, complex backgrounds, multilingual, handwriting.

> **📊 Planned diagram:** OCR pipeline — image → text detection (bounding polygons) → perspective rectification → text recognition (character sequence) → structured output.

## Text Detection

> **Planned content:** EAST: FCN predicting per-pixel text score + bounding box. DB (Differentiable Binarization): predict probability map + threshold map → binary map for text regions. PSENet, PAN: progressive scale expansion. PANNet. Polygon vs. quadrilateral detection. NMS for text proposals.

> **📊 Planned diagram:** DB text detection — probability map + threshold map → binary mask → contour-based polygon extraction. Differentiable binarization formula.

## Text Recognition: CRNN + CTC

> **Planned content:** Word-level recognition. Input: rectified text crop `[1, 1, 32, W]`. CNN features → `[1, C, 1, W/4]` → squeeze → `[W/4, 1, C]` → BiLSTM → `[W/4, 1, num_chars]` → CTC decode. CTC (Connectionist Temporal Classification) loss: allows aligning variable-length sequences without explicit segmentation. Greedy CTC decoding, beam search CTC decoding.

> **📊 Planned diagram (flowchart):** CRNN architecture — CNN column features → LSTM sequence → CTC output. CTC alignment paths visualization showing many-to-one mapping.

> **📊 Planned diagram:** CTC decoding — showing valid paths in the lattice that collapse to the target string, blank character role.

$$\mathcal{L}_{CTC} = -\log p(y \mid x) = -\log \sum_{\pi \in \mathcal{B}^{-1}(y)} p(\pi \mid x)$$

## Attention-Based Recognition (ASTER, ABINet)

> **Planned content:** Attention mechanism for recognizing text without CTC. Rectification + sequence attention. Two-dimensional attention. Language model branch (ABINet): iterative correction using a language model on the recognized sequence.

## End-to-End OCR Systems

> **Planned content:** PGNet, ABCNet, MANGO: detect and recognize jointly in one forward pass. End-to-end training. Masked-attention for arbitrary-shape text.

## Document Understanding

> **Planned content:** Beyond raw text: understanding layout (tables, headings, columns). LayoutLM (text + layout position + vision). LayoutLMv2, LayoutLMv3 (unified text-layout-vision). Donut (Document Understanding Transformer): end-to-end, no OCR engine needed. Table structure recognition (TATR).

> **📊 Planned diagram:** LayoutLMv3 input — tokens with 2D position embeddings and vision patches, all jointly encoded.

## Evaluation

> **Planned content:** Text detection: Hmean (harmonic mean of precision/recall with IoU > 0.5). Text recognition: word accuracy, edit distance, sequence accuracy. End-to-end: ICDAR competition metrics. Document: F1 on key-value extraction, table structure accuracy.

**Next: [Chapter 15 — Medical Imaging →](./15_medical_imaging.md)**

---
*Last updated: May 2026*
