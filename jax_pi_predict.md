# π0.5 JAX/OpenPI 推理过程记录

## 环境配置

由于 OpenPI 仓库依赖较复杂，建议使用 Python 3.11 环境：

```bash
# 创建虚拟环境
cd /home/ubuntu/stephen/01-code/openpi
uv venv .venv_jax2
source .venv_jax2/bin/activate

# 安装依赖
uv pip install -e packages/openpi-client/
uv sync  # 安装所有依赖（时间较长）
```

## 推理脚本

创建 `jax_predict.py`（位于 `/home/ubuntu/stephen/01-code/pi_operator/`）：

```python
import sys
import os

# 设置PYTHONPATH
sys.path.insert(0, "/home/ubuntu/stephen/01-code/openpi/src")
sys.path.insert(0, "/home/ubuntu/stephen/01-code/openpi/packages/openpi-client/src")

import jax
import numpy as np
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.policies import libero_policy

def load_model(checkpoint_path="/home/ubuntu/stephen/02-weight/pi05_libero"):
    config = _config.get_config("pi05_libero")
    policy = policy_config.create_trained_policy(config, checkpoint_path)
    return policy, config

def create_dummy_example():
    return libero_policy.make_libero_example()

def run_inference(policy, example):
    result = policy.infer(example)
    return result

if __name__ == "__main__":
    policy, config = load_model()
    example = create_dummy_example()
    result = run_inference(policy, example)
    print(f"Actions shape: {result['actions'].shape}")
```

## 依赖问题解决

### 问题 1: openpi_client 模块未安装
- 解决：安装 `packages/openpi-client`

### 问题 2: dm-tree 编译失败
- 原因：Python 3.13 缺少编译工具链
- 解决：使用 Python 3.11 虚拟环境

### 问题 3: lerobot API 不兼容
- 说明：OpenPI 使用旧版 lerobot API (lerobot.common.datasets)
- 解决：使用 PyTorch 版本模型进行推理（自动检测 model.safetensors）

## 实际运行结果

使用 PyTorch 框架成功运行推理：

```
Loading model from: /home/ubuntu/stephen/02-weight/pi05_libero
Config: pi05_libero
Model: Pi0Config(pi05=True, action_horizon=10, ...)
Model loaded successfully!
Example keys: dict_keys(['observation/state', 'observation/image', 'observation/wrist_image', 'prompt'])
Running inference...
Result keys: dict_keys(['actions'])
Actions shape: (10, 7)
✓ Inference successful!
```

## 结论

1. **OpenPI 仓库的 JAX 版本** 需要复杂的依赖链，包括特定版本的 lerobot
2. **实际推理使用的是 PyTorch 版本**：OpenPI 自动检测 `model.safetensors` 并使用 PyTorch 推理
3. **LeRobot 框架推理更成熟**：LeRobot 提供的 PyTorch 实现更加稳定

## 文件位置

- 模型权重: `/home/ubuntu/stephen/02-weight/pi05_libero/`
- OpenPI 仓库: `/home/ubuntu/stephen/01-code/openpi/`
- LeRobot 仓库: `/home/ubuntu/stephen/01-code/lerobot/`
- 推理脚本: `/home/ubuntu/stephen/01-code/pi_operator/jax_predict.py`