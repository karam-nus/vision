---
title: "Chapter 27 — Quantization"
---

[← Back to Table of Contents](./README.md)

# Chapter 27 — Quantization

> *"Quantization is the art of storing 32 bits of knowledge in 8 bits — or 4 bits, or fewer."*

## Quantization Fundamentals

> **Planned content:** Why quantize: 4× memory reduction (FP32 → INT8), 2–4× throughput on INT8 hardware (Tensor Cores, VNNI, NEON). Affine (asymmetric) quantization: `x_q = round((x / s) + z)`, `x_deq = s * (x_q - z)`. Symmetric: `z = 0`, simpler. Scale and zero-point. Per-tensor vs. per-channel quantization.

> **📊 Planned diagram:** Quantization pipeline — FP32 value range → clip to [α, β] → scale + round → INT8 integer → dequantize back → approximate FP32.

$$x_q = \text{clamp}\!\left(\text{round}\!\left(\frac{x}{s}\right) + z,\; q_{min},\, q_{max}\right), \qquad s = \frac{\beta - \alpha}{q_{max} - q_{min}}$$

## Post-Training Quantization (PTQ)

> **Planned content:** No retraining required. Calibration: run representative data through model to measure activation ranges (min/max, percentile, entropy/KLD). Weight quantization: per-channel symmetric. Activation quantization: per-tensor. SmoothQuant for challenging activations. Cross-layer equalization (CLE). Bias correction.

> **📊 Planned diagram:** PTQ pipeline — trained FP32 model → calibration dataset → compute scale/zero-point per layer → INT8 model → evaluate accuracy.

## Quantization-Aware Training (QAT)

> **Planned content:** Simulate quantization during training: insert fake quantization ops (FQ) in forward pass (round → gradient via STE). Model learns to tolerate quantization noise. Typically recovers 1-3% accuracy lost in PTQ. Straight-Through Estimator (STE): `∂L/∂x ≈ ∂L/∂x_q` (pass gradient through round op). Training cost: 10% of original training.

> **📊 Planned diagram:** QAT forward pass — FP32 weights → fake quantize → INT8 simulation → forward pass → loss → STE backward → update FP32 weights.

> **📊 Planned diagram:** STE gradient flow — showing where gradient would be zero (round op) but STE approximates it as 1 in the range.

```python
import torch.quantization as tq

# QAT preparation: insert FakeQuantize observers
model.train()
model.qconfig = tq.get_default_qat_qconfig('fbgemm')  # x86 INT8
model_prepared = tq.prepare_qat(model)

# Train for a few epochs with fake quantization
# ...

# Convert to real INT8 model
model_int8 = tq.convert(model_prepared.eval())
```

## Mixed Precision and Bit Widths

> **Planned content:** Not all layers need the same precision. First and last layers: typically stay FP32 (high sensitivity). INT8 for most layers. INT4 or W4A8 for extreme compression. HAWQv2, APQ: hardware-aware mixed precision search. FP8 training (H100+).

> **📊 Planned diagram:** Sensitivity analysis — accuracy drop vs. bit width for each layer, guiding mixed precision assignment.

## W8A8, W4A8, W4A16 Schemes

> **Planned content:** W8A8: INT8 weights + INT8 activations (GPU INT8 Tensor Cores). W4A16: INT4 weights + FP16 activations (memory bandwidth bound). W4A8: INT4 weights + INT8 activations (hardware-specific). GPTQ, AWQ style for W4A16.

## INT8 in Computer Vision

> **Planned content:** TensorRT INT8: post-training INT8 calibration with ICalibrator. OpenVINO POT (Post-Training Optimization Tool). PyTorch FX Graph Mode quantization. Accuracy comparison: ResNet-50 INT8 vs. FP32 (typically < 0.5% accuracy drop).

**Next: [Chapter 28 — Pruning & Sparsification →](./28_pruning_and_sparsification.md)**

---
*Last updated: May 2026*
