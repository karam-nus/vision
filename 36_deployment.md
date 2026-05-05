---
title: "Chapter 36 — Deployment"
---

[← Back to Table of Contents](./README.md)

# Chapter 36 — Deployment

> *"A model that isn't deployed doesn't exist. Deployment is where ML meets reality."*

## Model Export Pipeline

> **Planned content:** PyTorch → TorchScript (graph capture via tracing or scripting) → ONNX → TensorRT/.tflite/.mlmodel. Common gotchas: dynamic shapes, custom ops, control flow. Testing exported model vs. PyTorch model numerically.

> **📊 Planned diagram (flowchart):** Model export decision tree — server GPU → TensorRT; Intel CPU → OpenVINO; Mobile → TFLite (Android) / CoreML (iOS); Jetson → TensorRT; FPGA → custom.

## Serving Infrastructure

> **Planned content:** Triton Inference Server (NVIDIA): multi-model server, backends (TensorRT, ONNX, PyTorch, TF), dynamic batching, streaming. TorchServe: PyTorch-native model serving. FastAPI + uvicorn for custom REST endpoints. gRPC for high-throughput service. Model ensembles (preprocessing + model + postprocessing as pipeline).

> **📊 Planned diagram:** Triton serving architecture — client request → Triton server → backend (TensorRT engine) → response. Dynamic batching accumulation illustration.

## Edge Deployment

> **Planned content:** Jetson Orin: NVIDIA Xavier + GPU + DLA, TensorRT optimized, 275 TOPS. Raspberry Pi 4 + Coral TPU: EdgeTPU INT8. Qualcomm NPU: Snapdragon AI Engine, SNPE SDK. Rockchip RK3588: RKNN toolkit. Apple M-series ANE via CoreML. Power consumption vs. throughput tradeoffs.

> **📊 Planned diagram:** Edge device comparison — TOPS, memory bandwidth, power (W), supported precisions, SDK ecosystem.

## Latency vs. Throughput

> **Planned content:** Latency: time for single request. Throughput: requests per second at sustained load. Little's Law: throughput = N/latency (N = concurrent requests). Batch size effect: latency increases with batch, throughput increases and saturates. SLA-based optimization: optimize throughput subject to p99 latency < X ms.

> **📊 Planned diagram:** Latency-throughput tradeoff curve — showing how batching affects both, with optimal operating point for different SLAs.

## Preprocessing and Postprocessing in Pipeline

> **Planned content:** Don't neglect preprocessing latency: image decode (JPEG) + resize + normalize can be 30-50% of total latency. NVJPEG for GPU-accelerated JPEG decode. TensorRT preprocessing via CUDA kernels. Postprocessing (NMS, keypoint decode) also needs optimization.

## Model Versioning and A/B Testing

> **Planned content:** Model registry (MLflow, W&B Artifacts). Versioning strategy. Shadow mode: run new model in parallel, compare outputs. Canary deployment: 5% traffic → validate → 100%. Feature flags. Rollback procedures. Monitoring in production.

## Monitoring in Production

> **Planned content:** Data drift: input distribution shifts over time (new camera, lighting changes, season). Concept drift: label distribution changes. Performance monitoring: latency, throughput, error rate. Accuracy monitoring: human spot-check or auto-evaluation. Alerting and automated retraining triggers.

> **📊 Planned diagram:** Production monitoring dashboard — input feature distribution over time, model confidence distribution, latency p50/p95/p99.

**Next: [Chapter 37 — Ecosystem & Libraries →](./37_ecosystem_and_libraries.md)**

---
*Last updated: May 2026*
