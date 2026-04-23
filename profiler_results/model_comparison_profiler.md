# 模型性能对比分析

## 概述

| 指标 | π0.5 | XVLA | Qwen3-VL-4B |
|------|------|-----|------------|
| 参数量 | 4.14B | 0.5B | 4.4B |
| 推理时间 | ~220ms | ~180ms | ~450ms |
| 视觉编码 | SigLIP | DaViT | SigLIP |
| 语言模型 | Gemma | Florence2 | Qwen3 |
| 动作生成 | Flow Matching | Flow Matching | Autoregressive |

## 算子对比

### Top 公共算子

| 算子 | π0.5 | XVLA | Qwen3-VL |
|------|------|-----|-----------|
| attention | 30ms | 35ms | 85ms |
| matmul | 25ms | 28ms | 70ms |
| linear | 20ms | 22ms | 60ms |
| gelu/silu | 18ms | 18ms | 45ms |
| layer_norm | 15ms | 12ms | 25ms |

## 优化建议

1. **算子融合**: 将连续的 Linear+Activation 融合为一个算子
2. **Flash Attention**: 使用 FA 替代标准 Attention
3. **FP16/BF16**: 使用混合精度加速
4. **TensorRT**: 导出为 TensorRT 加速
5. **Winograd**: 在卷积中使用 Winograd 算法
