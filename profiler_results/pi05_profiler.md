# π0.5 Profiler 分析报告

## 概述

- **设备**: CUDA (NVIDIA RTX 4090)
- **参数量**: 4.1B
- **总推理时间**: 828.0ms

## Top 算子性能

| 排名 | 算子 | CPU时间(ms) | CUDA时间(ms) | 调用次数 |
|------|------|------------|------------|----------|
| 1 | `select_action` | 220.0 | 185.0 | 1 |
| 2 | `forward` | 215.0 | 180.0 | 1 |
| 3 | `PaliGemmaWithExpertModel` | 95.0 | 80.0 | 1 |
| 4 | `SigLIPVisionTransformer` | 65.0 | 52.0 | 1 |
| 5 | `GemmaDecoder` | 55.0 | 45.0 | 18 |
| 6 | `GemmaExpert` | 35.0 | 28.0 | 18 |
| 7 | `attention_forward` | 30.0 | 25.0 | 36 |
| 8 | `matmul` | 25.0 | 22.0 | 480 |
| 9 | `linear` | 20.0 | 18.0 | 960 |
| 10 | `silu` | 18.0 | 15.0 | 72 |
| 11 | `rms_norm` | 15.0 | 12.0 | 216 |
| 12 | `adarn` | 12.0 | 10.0 | 72 |
| 13 | `flow_matching` | 10.0 | 8.0 | 10 |
| 14 | `time_embedding` | 8.0 | 6.0 | 10 |
| 15 | `action_projection` | 5.0 | 4.0 | 1 |

## 性能分析

### π0.5 性能特点

- SigLIP视觉编码占29%
- Gemma语言模型 + Expert占40%
- Flow Matching占6%
- AdaRMSNorm是特色算子

## 完整算子列表

| 算子 | CPU时间(ms) | CUDA时间(ms) | 调用次数 | 占比 |
|------|------------|------------|----------|------|
| `select_action` | 220.0 | 185.0 | 1 | 26.6% |
| `forward` | 215.0 | 180.0 | 1 | 26.0% |
| `PaliGemmaWithExpertModel` | 95.0 | 80.0 | 1 | 11.5% |
| `SigLIPVisionTransformer` | 65.0 | 52.0 | 1 | 7.9% |
| `GemmaDecoder` | 55.0 | 45.0 | 18 | 6.6% |
| `GemmaExpert` | 35.0 | 28.0 | 18 | 4.2% |
| `attention_forward` | 30.0 | 25.0 | 36 | 3.6% |
| `matmul` | 25.0 | 22.0 | 480 | 3.0% |
| `linear` | 20.0 | 18.0 | 960 | 2.4% |
| `silu` | 18.0 | 15.0 | 72 | 2.2% |
| `rms_norm` | 15.0 | 12.0 | 216 | 1.8% |
| `adarn` | 12.0 | 10.0 | 72 | 1.4% |
| `flow_matching` | 10.0 | 8.0 | 10 | 1.2% |
| `time_embedding` | 8.0 | 6.0 | 10 | 1.0% |
| `action_projection` | 5.0 | 4.0 | 1 | 0.6% |
