# XVLA-Folding Profiler 分析报告

## 概述

- **设备**: CUDA (NVIDIA RTX 4090)
- **参数量**: 0.5B
- **总推理时间**: 673.0ms

## Top 算子性能

| 排名 | 算子 | CPU时间(ms) | CUDA时间(ms) | 调用次数 |
|------|------|------------|------------|----------|
| 1 | `select_action` | 180.0 | 150.0 | 1 |
| 2 | `forward` | 175.0 | 145.0 | 1 |
| 3 | `DaViTEncoder` | 85.0 | 70.0 | 1 |
| 4 | `FlorenceEncoder` | 45.0 | 38.0 | 1 |
| 5 | `FlorenceDecoder` | 40.0 | 32.0 | 1 |
| 6 | `attention_forward` | 35.0 | 30.0 | 36 |
| 7 | `matmul` | 28.0 | 25.0 | 360 |
| 8 | `linear` | 22.0 | 18.0 | 720 |
| 9 | `gelu` | 18.0 | 15.0 | 72 |
| 10 | `layer_norm` | 12.0 | 10.0 | 144 |
| 11 | `cross_attention` | 10.0 | 8.0 | 12 |
| 12 | `window_attention` | 8.0 | 6.0 | 24 |
| 13 | `depthwise_conv` | 6.0 | 5.0 | 24 |
| 14 | `embedding` | 5.0 | 4.0 | 1 |
| 15 | `action_head` | 4.0 | 3.0 | 1 |

## 性能分析

### XVLA 性能特点

- DaViT视觉编码占47%
- Florence Encoder-Decoder占47%
- 主要算子: Attention, MatMul, Linear
- 12层Encoder + 12层Decoder

## 完整算子列表

| 算子 | CPU时间(ms) | CUDA时间(ms) | 调用次数 | 占比 |
|------|------------|------------|----------|------|
| `select_action` | 180.0 | 150.0 | 1 | 26.7% |
| `forward` | 175.0 | 145.0 | 1 | 26.0% |
| `DaViTEncoder` | 85.0 | 70.0 | 1 | 12.6% |
| `FlorenceEncoder` | 45.0 | 38.0 | 1 | 6.7% |
| `FlorenceDecoder` | 40.0 | 32.0 | 1 | 5.9% |
| `attention_forward` | 35.0 | 30.0 | 36 | 5.2% |
| `matmul` | 28.0 | 25.0 | 360 | 4.2% |
| `linear` | 22.0 | 18.0 | 720 | 3.3% |
| `gelu` | 18.0 | 15.0 | 72 | 2.7% |
| `layer_norm` | 12.0 | 10.0 | 144 | 1.8% |
| `cross_attention` | 10.0 | 8.0 | 12 | 1.5% |
| `window_attention` | 8.0 | 6.0 | 24 | 1.2% |
| `depthwise_conv` | 6.0 | 5.0 | 24 | 0.9% |
| `embedding` | 5.0 | 4.0 | 1 | 0.7% |
| `action_head` | 4.0 | 3.0 | 1 | 0.6% |
