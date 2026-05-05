---
title: "Chapter 26 — Optimization Overview"
---

[← Back to Table of Contents](./README.md)

# Chapter 26 — Model Optimization Overview

> *"The best model is not the most accurate one — it's the one that runs on your hardware within your latency budget with acceptable accuracy."*

## Why Optimize?

> **Planned content:** Research vs. deployment requirements. A ViT-H has 632M parameters — fine for GPU inference, impractical for a smartphone. Optimization bridges this gap. The three axes: accuracy, latency, model size. The tradeoff triangle.

> **📊 Planned diagram:** The optimization tradeoff triangle — accuracy vs. latency vs. model size, with different techniques' positions labeled.

## The Optimization Landscape

> **Planned content:** Four main families: (1) Quantization — reduce numerical precision. (2) Pruning/Sparsification — remove weights or structure. (3) Knowledge Distillation — compress knowledge into smaller model. (4) Neural Architecture Search — find efficient architectures. (5) Inference optimization — graph compilation, operator fusion.

> **📊 Planned diagram:** Optimization family tree — taxonomy with representative techniques per family.

## Hardware Targets

> **Planned content:** Server GPU (A100, H100): FP16/BF16, INT8, FP8. Edge GPU (Jetson Orin): INT8, INT4. Mobile GPU (Adreno, Apple Neural Engine): FP16, INT8. CPU (x86, ARM): INT8, sparse. FPGA: custom bitwidth. Each target has different optimal optimization strategy.

> **📊 Planned table:** Hardware targets × optimization compatibility matrix — which techniques work well on which hardware.

## Profiling and Bottleneck Analysis

> **Planned content:** Before optimizing, profile! Latency breakdown: memory-bound vs. compute-bound ops. PyTorch Profiler. Roofline model for CV models. Memory bandwidth vs. arithmetic intensity. FLOPs vs. latency (why more FLOPs ≠ more latency). Batch size effects. Layer-wise latency analysis.

> **📊 Planned diagram:** Roofline model for a ResNet-50 layer — plotting arithmetic intensity vs. performance, roof = max compute or max bandwidth.

## Optimization Pipeline

> **Planned content:** Typical pipeline: train full model → profile → quantize PTQ → evaluate accuracy → if not enough: QAT → prune → distill smaller model → export to ONNX/TensorRT → benchmark on target hardware.

> **📊 Planned diagram (flowchart):** Model optimization decision tree — start with PTQ, evaluate accuracy drop, if too large → try QAT → if still not enough → distillation or architecture replacement.

## FLOP Counting

> **Planned content:** FLOPs (Floating Point Operations): 2 × M × N × K for matmul `[M, K] × [K, N]`. FLOPs for Conv2d: `2 × C_in × C_out × k² × H_out × W_out`. Tools: thop, fvcore, torchinfo. MACs vs. FLOPs (1 MAC = 2 FLOPs). Why FLOPs are a poor proxy for latency.

```python
from thop import profile
import torch
from torchvision.models import resnet50

model = resnet50()
x = torch.randn(1, 3, 224, 224)  # [1, 3, 224, 224]
macs, params = profile(model, inputs=(x,))
print(f"MACs: {macs/1e9:.2f}G, Params: {params/1e6:.2f}M")
# MACs: 4.09G, Params: 25.56M
```

**Next: [Chapter 27 — Quantization →](./27_quantization.md)**

---
*Last updated: May 2026*
