# π0.5 模型算子报告

## 1. 模型概述

π0.5 (PI05) 是 Physical Intelligence 开发的 VLA (Vision-Language-Action) 模型，核心创新是**异构数据联合训练**，实现开放世界泛化能力。

- **论文**: https://arxiv.org/abs/2504.16054
- **框架**: LeRobot (HuggingFace)
- **模型路径**: `/home/ubuntu/stephen/02-weight/pi05_libero/`
- **Tokenizer 路径**: `/home/ubuntu/stephen/02-weight/paligemma-3b-pt-224/`

---

## 2. 算子架构总览

### 2.1 输入输出

| 输入 | 形状 | 说明 |
|------|------|------|
| `observation.images.image` | (B, 3, 256, 256) | 相机1图像 |
| `observation.images.image2` | (B, 3, 256, 256) | 相机2图像 |
| `observation.state` | (B, 8) | 机器人状态 |
| `task` | str | 语言指令 |

| 输出 | 形状 | 说明 |
|------|------|------|
| `action` | (B, 7) | 机器人关节动作 |

### 2.2 核心组件

```
输入 → [视觉编码器] → [语言编码器] → [ Expert Action Decoder] → 动作
         (SigLIP)      (Gemma)      (Flow Matching)
```

---

## 3. 核心算子详解

### 3.1 PaliGemmaWithExpertModel

**功能**: 视觉-语言联合编码器，含动作 expert

**组成**:

| 组件 | 类型 | 参数量 |
|------|------|--------|
| `vision_tower` | SigLIP ViT | ~400M |
| `language_model` | Gemma 2B | ~2B |
| `gemma_expert` | Gemma 300M | ~300M |

**代码位置**: `lerobot/src/lerobot/policies/pi_gemma.py`

**描述**:
- `vision_tower`: 处理图像输入，输出视觉特征
- `language_model`: Gemma transformer，处理文本token
- `gemma_expert`: 动作预测分支，含 AdaRMSNorm

### 3.2 PiGemmaRMSNorm

**功能**: 自适应 RMS 归一化 (AdaRMS)

**代码位置**: `lerobot/src/lerobot/policies/pi_gemma.py:85`

```python
class PiGemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, cond_dim: int | None = None):
        # cond_dim=None: 标准 RMSNorm
        # cond_dim!=None: 条件归一化 (scale, shift, gate)
```

**参数量**:

| 模式 | 参数量 |
|------|--------|
| 标准 | dim (weight) |
| AdaRMS | dim × 3 (linear) |

### 3.3 Flow Matching Action Head

**功能**: 基于 Flow Matching 的动作生成

**代码位置**: `lerobot/src/lerobot/policies/pi05/modeling_pi05.py:551`

**组成**:

| 算子 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `action_in_proj` | (B, 50, 8) | (B, 50, 2048) | 动作投影 |
| `time_mlp_in` | (B, 2048) | (B, 2048) | 时间嵌入 |
| `time_mlp_out` | (B, 2048) | (B, 2048) | 时间 MLP |
| `action_out_proj` | (B, 50, 2048) | (B, 50, 8) | 输出投影 |

**Flow Matching 损失**:

```
x_t = t * noise + (1-t) * actions   # 噪声调度
v_t = model(x_t, t)             # 预测速度
loss = MSE(noise - actions, v_t)   # Flow 目标
```

### 3.4 图像预处理算子

**代码位置**: `lerobot/src/lerobot/policies/pi05/modeling_pi05.py:152`

```python
def resize_with_pad_torch(images, height, width, mode="bilinear"):
    # 1. 按比例缩放图像
    # 2. 填充到目标尺寸
```

### 3.5 位置编码

**Sinusoidal Pos Embedding** (`modeling_pi05.py:81`):

```python
def create_sinusoidal_pos_embedding(time, dimension, min_period, max_period):
    # 输出一组 sin-cos 位置编码
    # 用于动作时间步嵌入
```

**RoPE (旋转位置编码)**:

- 继承自 Gemma 模型
- 用于自注意力机制

---

## 4. 训练算子配置

### 4.1 PI05Config 参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `paligemma_variant` | `gemma_2b` | VLM 骨干 |
| `action_expert_variant` | `gemma_300m` | 动作 Expert |
| `chunk_size` | 50 | 动作预测步数 |
| `n_action_steps` | 50 | 执行���数 |
| `num_inference_steps` | 10 | 推理采样步数 |
| `image_resolution` | (224, 224) | 图像尺寸 |
| `optimizer_lr` | 2.5e-5 | 学习率 |
| `gradient_checkpointing` | False | 梯度 checkpoint |

### 4.2 归一化模式

| 特征类型 | 归一化方式 |
|----------|------------|
| VISUAL | IDENTITY (无) |
| STATE | QUANTILES |
| ACTION | QUANTILES |

---

## 5. 推理流程

### 5.1 前向传播

```
1. embed_prefix(images, tokens)
   ↓
   图像 → SigLIP → 视觉embed
   文本 → Gemma → 语言embed
   ↓
2. embed_suffix(noise, timestep)
   ↓
   噪声 + 时间 → 动作embed + 时间embed
   ↓
3. PaliGemmaForward
   ↓
   [视觉emb, 语言emb] + [动作emb] → Transformer
   ↓
4. action_out_proj
   ↓
   输出动作 velocity
```

### 5.2 采样流程 (Flow Matching)

```
1. x_0 = noise (随机初始化)
2. for step in range(num_steps):
      t = 1 - step/num_steps
      v = denoise(x_t, t)
      x_{t+1} = x_t + dt * v
3. return x_last
```

---

## 6. 参数量统计

### 6.1 各组件参数量

| 组件 | 参数量 | 占比 |
|------|--------|------|
| SigLIP (vision) | ~400M | ~14% |
| Gemma 2B (language) | ~2B | ~69% |
| Gemma 300M (expert) | ~300M | ~10% |
| 动作头 | ~50M | ~2% |
| 其他投影 | ~10M | ~1% |
| **总计** | **~2.8B** | 100% |

### 6.2 模型文件

```
pi05_libero/
├── model.safetensors    # 13.5GB (权重)
├── config.json         # 配置
├── policy_preprocessor.json
└── policy_postprocessor.json
```

---

## 7. 常见 LLM 算子对比

### 7.1 标准 LLM 算子

| 算子 | 功能 | 参数量公式 |
|------|------|----------|
| Embedding | token→向量 | V × d |
| Attention | 自注意力 | 4d² |
| FFN/MLP | 前馈网络 | 2 × d × 4d |
| LayerNorm | 归一化 | d |
| RoPE | 位置编码 | 无参数 |
| LM Head | 词表预测 | V × d |

### 7.2 π0.5 特有算子

| 算子 | 功能 | 说明 |
|------|------|------|
| PiGemmaRMSNorm | 自适应归一化 | 条件 RMSNorm |
| Flow Matching | 动作生成 | 连续动作预测 |
| SinusoidalPosEmb | 时间编码 | 动作时间步 |
| Action Projection | 动作投影 | 状态→动作 |

---

## 8. 参考资料

- 模型代码: `/home/ubuntu/stephen/01-code/lerobot/src/lerobot/policies/pi05/`
- 配置文件: `/home/ubuntu/stephen/01-code/lerobot/src/lerobot/policies/pi05/configuration_pi05.py`
- LeRobot: https://github.com/huggingface/lerobot
- OpenPI: https://github.com/Physical-Intelligence/openpi