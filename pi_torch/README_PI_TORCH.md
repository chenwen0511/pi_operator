# π0.5 VLA Model Operators

本仓库包含 π0.5 (PI05) 视觉语言动作模型的算子分析和推理代码，基于 [LeRobot](https://github.com/huggingface/lerobot) 框架。

## 模型概述

π0.5 是 Physical Intelligence 开发的 VLA (Vision-Language-Action) 模型，通过异构数据联合训练实现开放世界泛化能力。

- **论文**: [π0.5: A Vision-Language-Action Model with Open-World Generalization](https://arxiv.org/abs/2504.16054)
- **官方开源**: [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
- **框架**: [LeRobot](https://github.com/huggingface/lerobot)

## 核心算子

| 算子 | 功能 | 说明 |
|------|------|------|
| **SigLIP Vision** | 视觉编码器 | 处理相机图像输入 (~400M参数) |
| **Gemma Language** | 语言编码器 | 处理文本指令 (~2B参数) |
| **Flow Matching** | 动作生成 | 基于连续动作预测 |
| **PiGemmaRMSNorm** | 自适应归一化 | 条件 RMSNorm |

详见 [PI_Operator_report.md](./PI_Operator_report.md)

## 环境准备

### 1. 安装依赖

```bash
# 克隆 LeRobot
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e . --no-deps
pip install torch transformers safetensors gymnasium -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 下载模型

```bash
# 下载 π0.5 模型 (需要申请访问权限)
modelscope download --model lerobot/pi05_libero --local_dir /path/to/pi05_libero/

# 下载 Paligemma Tokenizer (用于本地)
modelscope download --model AI-ModelScope/paligemma-3b-pt-224 --local_dir /path/to/paligemma-3b-pt-224/
```

## 使用方法

### 推理示例

```python
import torch
from lerobot.policies.pi05 import PI05Policy
from lerobot.policies.factory import make_pre_post_processors

# 模型路径
model_path = "/your/path/to/pi05_libero"
tokenizer_path = "/your/path/to/paligemma-3b-pt-224"

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = PI05Policy.from_pretrained(model_path).to(device).eval()

# 创建预处理器
preprocess, postprocess = make_pre_post_processors(
    policy.config,
    model_path,
    preprocessor_overrides={
        "device_processor": {"device": str(device)},
        "tokenizer_processor": {"tokenizer_name": tokenizer_path},
    },
)

# 准备输入
example = {
    "observation.images.image": torch.randn(3, 256, 256),
    "observation.images.image2": torch.randn(3, 256, 256),
    "observation.state": torch.randn(8),
    "task": "pick up the object",
}

# 推理
batch = preprocess(example)
batch = {k: v.to(device) for k, v in batch.items() if hasattr(v, 'to')}

with torch.inference_mode():
    action = policy.select_action(batch)
    action = postprocess(action)

print(f"Action: {action.shape}")  # (1, 7)
```

### 使用本仓库脚本

```bash
# 修改 predict.py 中的模型路径后运行
python predict.py
```

## 项目结构

```
pi_operator/
├── predict.py              # 推理脚本
├── PI_Operator_report.md # 算子详细报告
└── README.md            # 本文件
```

## 模型文件

需要下载的模型文件：

| 文件 | 大小 | 说明 |
|------|------|------|
| `pi05_libero/` | ~13.5GB | π0.5 权重 |
| `paligemma-3b-pt-224/` | ~11GB | 语言Tokenizer |

## 硬件要求

| 模式 | 显存 | GPU |
|------|------|-----|
| 推理 | >8GB | RTX 4090 ✓ |
| LoRA微调 | >22GB | RTX 4090 ✓ |
| 全参数微调 | >70GB | A100/H100 |

## 参考资料

- [π0.5 论文](https://arxiv.org/abs/2504.16054)
- [OpenPI 官方仓库](https://github.com/Physical-Intelligence/openpi)
- [LeRobot 文档](https://huggingface.co/docs/lerobot)
- [Physical Intelligence 官网](https://www.pi.website)
