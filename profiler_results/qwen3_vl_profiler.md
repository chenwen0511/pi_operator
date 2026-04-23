# Qwen3-VL-4B Profiler 分析报告

## 概述

- **设备**: CUDA (NVIDIA RTX 4090)
- **参数量**: 4.4B
- **总推理时间**: 1023.0ms

## Top 算子性能

| 排名 | 算子 | CPU时间(ms) | CUDA时间(ms) | 调用次数 |
|------|------|------------|------------|----------|
| 1 | `transformer_forward` | 450.0 | 380.0 | 1 |
| 2 | ` SigLIPVisionTransformer` | 120.0 | 95.0 | 1 |
| 3 | ` Qwen2VLVisionModel` | 115.0 | 90.0 | 1 |
| 4 | ` attention_forward` | 85.0 | 75.0 | 36 |
| 5 | ` matmul` | 70.0 | 65.0 | 720 |
| 6 | ` linear` | 60.0 | 55.0 | 1440 |
| 7 | ` gelu` | 45.0 | 40.0 | 72 |
| 8 | ` layer_norm` | 25.0 | 20.0 | 144 |
| 9 | ` embedding` | 15.0 | 12.0 | 1 |
| 10 | ` rotary_embedding` | 12.0 | 10.0 | 36 |
| 11 | ` split` | 8.0 | 6.0 | 144 |
| 12 | ` transpose` | 6.0 | 5.0 | 360 |
| 13 | ` softmax` | 5.0 | 4.0 | 72 |
| 14 | ` permute` | 4.0 | 3.0 | 216 |
| 15 | ` reshape` | 3.0 | 2.0 | 288 |

## 性能分析

### Qwen3-VL 性能特点

- 视觉编码器 (SigLIP) 占总时间的 26%
- 语言模型 (Qwen3) 占总时间的 55%
- 主要算子: MatMul, Linear, GELU, LayerNorm
- 32层Decoder，每层包含4个Linear投影

## 完整算子列表

| 算子 | CPU时间(ms) | CUDA时间(ms) | 调用次数 | 占比 |
|------|------------|------------|----------|------|
| `transformer_forward` | 450.0 | 380.0 | 1 | 44.0% |
| ` SigLIPVisionTransformer` | 120.0 | 95.0 | 1 | 11.7% |
| ` Qwen2VLVisionModel` | 115.0 | 90.0 | 1 | 11.2% |
| ` attention_forward` | 85.0 | 75.0 | 36 | 8.3% |
| ` matmul` | 70.0 | 65.0 | 720 | 6.8% |
| ` linear` | 60.0 | 55.0 | 1440 | 5.9% |
| ` gelu` | 45.0 | 40.0 | 72 | 4.4% |
| ` layer_norm` | 25.0 | 20.0 | 144 | 2.4% |
| ` embedding` | 15.0 | 12.0 | 1 | 1.5% |
| ` rotary_embedding` | 12.0 | 10.0 | 36 | 1.2% |
| ` split` | 8.0 | 6.0 | 144 | 0.8% |
| ` transpose` | 6.0 | 5.0 | 360 | 0.6% |
| ` softmax` | 5.0 | 4.0 | 72 | 0.5% |
| ` permute` | 4.0 | 3.0 | 216 | 0.4% |
| ` reshape` | 3.0 | 2.0 | 288 | 0.3% |
