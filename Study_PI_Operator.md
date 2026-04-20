# π0.5 模型算子介绍与操作步骤

## 模型概述

π0.5 是 Physical Intelligence 开发的 VLA (Vision-Language-Action) 模型，核心创新是**异构数据联合训练 (co-training)**，实现开放世界泛化能力，可在未见过的家庭环境中执行清理厨房、卧室等长时程任务。

官方论文：https://arxiv.org/abs/2504.16054
官方博客：https://www.pi.website/blog/pi05
开源仓库：https://github.com/Physical-Intelligence/openpi

### 核心算子架构

| 算子 | 功能 | 基础模型 |
|------|------|----------|
| **视觉编码器** | 处理相机图像输入 | SigLIP (SO-400M) |
| **语言编码器** | 处理文本指令 | Gemma-2B |
| **高层动作解码器** | 将复杂指令转换为语义动作描述 | Gemma decoder head |
| **动作专家** (Flow Matching) | 生成底层机器人关节动作 | Flow Matching Loss |

### 数据来源

- 多种机器人演示数据
- 高层语义预测任务
- 多模态网页数据
- 目标检测数据

---

## 可操作步骤

### 1. 环境准备

```bash
# 克隆仓库（含子模块）
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
cd openpi

# 安装依赖（使用 uv）
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### 2. 下载模型检查点

```python
from openpi.shared import download

# 下载基础模型
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_base")

# 下载已微调模型（可选）
# pi05_droid: 适合 DROID 平台
# pi05_libero: 适合 LIBERO 基准
```

### 3. 运行推理

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")

policy = policy_config.create_trained_policy(config, checkpoint_dir)

example = {
    "observation/exterior_image_1_left": ...,
    "observation/wrist_image_left": ...,
    "prompt": "pick up the fork"
}
action_chunk = policy.infer(example)["actions"]
```

### 4. 微调训练

```bash
# 计算归一化统计量
uv run scripts/compute_norm_stats.py --config-name pi05_libero

# 开始训练
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero \
    --exp-name=my_experiment --overwrite
```

### 5. PyTorch 支持（2025年9月新增）

```bash
# 应用 transformers 补丁
cp -r ./src/openpi/models_pytorch/transformers_replace/* \
    .venv/lib/python3.11/site-packages/transformers/

# 转换 JAX 模型到 PyTorch
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /path/to/jax/checkpoint \
    --config_name pi05_droid \
    --output_path /path/to/pytorch/checkpoint
```

---

## 硬件要求

| 模式 | 显存需求 | 适用 GPU |
|------|----------|---------|
| 推理 | >8 GB | RTX 4090 ✓ |
| LoRA 微调 | >22.5 GB | RTX 4090 ✓ |
| 全参数微调 | >70 GB | A100/H100 |

你的 RTX 4090 可以完成推理和 LoRA 微调。

---

## 快速验证

```bash
# 无机器人测试推理
uv run examples/simple_client/README.md
```

---

## 参考资料

- 论文：https://arxiv.org/abs/2504.16054
- 官方博客：https://www.pi.website/blog/pi05
- GitHub：https://github.com/Physical-Intelligence/openpi
- PyTorch 实现：需应用 transformers 补丁