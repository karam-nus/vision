---
title: "Chapter 31 — Efficient Inference"
---

[← Back to Table of Contents](./README.md)

# Chapter 31 — Efficient Inference

> *"A model that runs in 100ms on GPU is useless on a Raspberry Pi. Efficient inference is about bridging training and deployment realities."*

## Graph-Level Optimization

> **Planned content:** Operator fusion: fuse Conv + BN + ReLU into a single kernel (eliminates intermediate memory writes). Constant folding: pre-compute BN parameters into conv weights. Dead code elimination. Layout optimization (NCHW → NHWC for ARM). Graph partitioning. Kernel auto-tuning (TVM, Halide).

> **📊 Planned diagram:** Operator fusion — Conv → BN → ReLU without fusion (3 memory round-trips) vs. fused (1 kernel, 1 memory round-trip).

## ONNX: Universal Model Exchange

> **Planned content:** Open Neural Network Exchange: IR for ML models. Export from PyTorch/TensorFlow → optimize → deploy on any runtime. ONNX operators. Dynamic vs. static shapes. onnxruntime: optimized ONNX inference (CPU, GPU, Edge). ONNX simplifier.

```python
import torch
import torch.onnx

model = ...  # your trained PyTorch model
dummy_input = torch.randn(1, 3, 224, 224)  # [B, C, H, W]

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=17,
    input_names=["input"],     # name inputs
    output_names=["output"],   # name outputs
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
```

## TensorRT: NVIDIA GPU Optimization

> **Planned content:** TensorRT: optimize ONNX models for NVIDIA GPUs. Layer fusion, kernel auto-selection, precision calibration (FP16, INT8). Build engine from ONNX → serialize as .plan file → deserialize and run. Builder API. ICalibrator for INT8. Layer-by-layer timing profiles. Triton Inference Server for serving.

> **📊 Planned diagram (flowchart):** TensorRT workflow — ONNX model → TensorRT builder → optimized engine `.plan` → runtime inference with tensors `[B, C, H, W]`.

```python
import tensorrt as trt
import numpy as np

# Build TensorRT engine from ONNX
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser  = trt.OnnxParser(network, logger)
with open("model.onnx", "rb") as f:
    parser.parse(f.read())

config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)       # enable FP16

engine = builder.build_serialized_network(network, config)
```

## OpenVINO: Intel Optimization

> **Planned content:** Intel's inference toolkit: optimize for Intel CPUs, iGPUs, Movidius VPU, FPGA. Model Optimizer: convert ONNX/TF/PyTorch to IR (.xml + .bin). Post-Training Optimization Tool (POT) for INT8. OpenVINO Runtime for deployment. NNCF for QAT-ready models.

## TFLite and CoreML: Mobile Deployment

> **Planned content:** TFLite: TensorFlow Lite for Android/iOS/embedded. XNNPACK backend. INT8 quantization with representative dataset. Flatbuffer format. CoreML: Apple's on-device inference. ANE (Apple Neural Engine) acceleration. coremltools: convert PyTorch/ONNX → CoreML.

> **📊 Planned diagram:** Mobile deployment pipeline — PyTorch → ONNX → TFLite/CoreML → Android/iOS device inference.

## NCNN and MNN: Ultra-Lightweight Mobile

> **Planned content:** NCNN (Tencent): ARM-optimized C++ inference library. No dependencies. Supports many operators. Widely used in mobile vision apps. MNN (Alibaba): multi-backend mobile inference. Vulkan GPU support.

## Batching Strategies

> **Planned content:** Static batching: fixed batch size B. Dynamic batching (Triton): accumulate requests up to max_batch or timeout. Continuous batching for LLMs. Optimal batch size for throughput vs. latency tradeoff. Memory bandwidth bound vs. compute bound transition point.

> **📊 Planned diagram:** Throughput vs. latency vs. batch size curve — showing knee point where throughput saturates and latency inflects.

## Benchmark: Latency vs. Accuracy on Common Platforms

> **Planned content:** ResNet-50, MobileNetV3, EfficientNet-B0 on A100 GPU, Jetson Orin, iPhone 14 ANE, Raspberry Pi 4. Comparison of FP32, FP16, INT8. Tools: netron (visualization), trtexec (TensorRT benchmark), OpenVINO benchmark_app.

> **📊 Planned table:** Model × platform × precision → latency (ms), throughput (img/s), accuracy drop (%).

**Next: [Chapter 32 — Datasets →](./32_datasets.md)**

---
*Last updated: May 2026*
