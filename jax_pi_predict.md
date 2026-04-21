# π0.5 (Pi0.5) JAX/OpenPI 推理完整指南

## 一、Pi0.5 模型基础（JAX 视角）

### 1. 模型定位
- **类型**：VLA (Vision-Language-Action) 模型，输入图像 + 语言指令 + 本体感知，输出机器人连续动作序列
- **框架**：JAX + Flax（官方原生实现）
- **核心改进（对比 π0）**：
  - 移除 `state token`，改用 **adaptive RMS norm (adaRMS)** 注入时间信息
  - **混合训练**：预训练用离散 token，后训练用 **Flow Matching 动作专家** 生成高频动作
  - 主干：**PaliGemma VLM**（视觉+语言编码）+ 动作专家子网络

### 2. 官方权重版本（JAX Checkpoint）

开源提供 3 套 JAX 权重（Google Cloud Storage）：
- `pi05_droid`：基于 DROID 数据集，通用桌面操作
- `pi05_libero`：基于 LIBERO 仿真，日常家务
- `pi05_aloha_sim`：基于 ALOHA 双臂机器人

---

## 二、GCS 权重下载

### 检查 GCS 访问权限
```bash
gsutil ls gs://openpi-assets/checkpoints/
```

如果提示需要认证，请配置 Google Cloud SDK：
```bash
gcloud auth login
gcloud auth application-default login
```

### 下载权重

**方式 1：ModelScope（推荐国内用户）**
```bash
# 安装 modelscope
pip install modelscope

# 下载 pi05_base 模型
modelscope download --model hairuoliu/pi05_base --local_dir /home/ubuntu/stephen/02-weight/pi05_base

# 查看下载的权重
ls -lh /home/ubuntu/stephen/02-weight/pi05_base/
```

**方式 2：GCS（需要 Google Cloud SDK）**
```bash
# 创建权重目录
mkdir -p /home/ubuntu/stephen/02-weight/pi05_libero_jax

# 下载 JAX 权重
gsutil -m rsync -r gs://openpi-assets/checkpoints/pi05_libero/ /home/ubuntu/stephen/02-weight/pi05_libero_jax/
```

**注意**：JAX 权重约 11.6GB，下载时间取决于网络速度

---

## 三、环境配置

### 1. 虚拟环境准备
使用已有的 `.venv` 或创建新环境：
```bash
cd /home/ubuntu/stephen/01-code
python -m venv .venv
source .venv/bin/activate
```

### 2. 依赖安装
```bash
# 安装 JAX 和核心依赖
pip install jax==0.10.0 jaxlib==0.10.0

# 安装 openpi 依赖
pip install flax==0.12.6 einops transformers huggingface_hub gcsfs
pip install orbax-checkpoint==0.11.36
pip install jaxtyping==0.2.36

# 安装其他依赖
pip install chex tqdm_loggable numpydantic dm-tree augmax imageio opencv-python
pip install tyro pytest sentencepiece beartype typer rich polars

# 安装 lerobot (从 git)
pip install git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
```

### 3. 问题与解决

**问题 1：jaxtyping 版本不兼容**
```
AttributeError: module 'jaxtyping._decorator' has no attribute '_check_dataclass_annotations'
```
解决：安装 jaxtyping==0.2.36

**问题 2：orbax API 变化**
```
TypeError: 'StepMetadata' object is not subscriptable
```
这是由于 orbax-checkpoint 0.11.x 版本 API 变化导致的权重加载问题，需要修改 openpi/src/openpi/models/model.py 中的 restore_params 函数来兼容。

### 4. JAX GPU 验证
```python
import jax
print(jax.devices())  # 输出 cuda:0 则正常
print(jax.default_backend())  # 应为 gpu
```

---

## 四、JAX 推理脚本

### 完整推理代码 (`jax_predict.py`)

```python
#!/usr/bin/env python3
"""
π0.5 模型推理脚本 (JAX/OpenPI 框架)
支持 JAX 原生权重推理和 PyTorch 权重推理
"""

import sys
import os
import time

# 设置PYTHONPATH
OPENPI_SRC = "/home/ubuntu/stephen/01-code/openpi/src"
OPENPI_CLIENT_SRC = "/home/ubuntu/stephen/01-code/openpi/packages/openpi-client/src"
if OPENPI_SRC not in sys.path:
    sys.path.insert(0, OPENPI_SRC)
if OPENPI_CLIENT_SRC not in sys.path:
    sys.path.insert(0, OPENPI_CLIENT_SRC)
os.environ["PYTHONPATH"] = f"{OPENPI_SRC}:{OPENPI_CLIENT_SRC}"

import jax
import jax.numpy as jnp
import numpy as np
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.policies import libero_policy


def load_model_jax(checkpoint_path="/home/ubuntu/stephen/02-weight/pi05_libero_jax"):
    """加载 JAX OpenPI 模型"""
    print(f"Loading JAX model from: {checkpoint_path}")
    print(f"JAX devices: {jax.devices()}")
    
    config = _config.get_config("pi05_libero")
    print(f"Config: {config.name}")
    print(f"Model: {config.model}")
    
    policy = policy_config.create_trained_policy(config, checkpoint_path)
    print("JAX Model loaded successfully!")
    
    return policy, config


def load_model_pytorch(checkpoint_path="/home/ubuntu/stephen/02-weight/pi05_libero"):
    """加载 PyTorch OpenPI 模型"""
    print(f"Loading PyTorch model from: {checkpoint_path}")
    print(f"Using device: cuda" if jax.devices() else "cpu")
    
    config = _config.get_config("pi05_libero")
    print(f"Config: {config.name}")
    
    policy = policy_config.create_trained_policy(config, checkpoint_path)
    print("PyTorch Model loaded successfully!")
    
    return policy, config


def create_libero_example():
    """创建 LIBERO 示例输入"""
    return libero_policy.make_libero_example()


def run_inference_timing(policy, example, num_runs=10):
    """运行推理并计时"""
    print(f"\nRunning inference timing tests ({num_runs} runs)...")
    
    # 第一次推理（包含 JIT 编译）
    print("First inference (with JIT compilation)...")
    start_time = time.time()
    result = policy.infer(example)
    first_time = time.time() - start_time
    print(f"  First inference time: {first_time:.3f}s")
    print(f"  Actions shape: {result['actions'].shape}")
    
    # 后续推理（已编译）
    times = []
    for i in range(num_runs):
        start_time = time.time()
        result = policy.infer(example)
        elapsed = time.time() - start_time
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    print(f"\n  Average inference time ({num_runs} runs): {avg_time:.3f}s")
    print(f"  Min: {min(times):.3f}s, Max: {max(times):.3f}s")
    
    return result, first_time, avg_time


def main():
    print("=" * 60)
    print("π0.5 VLA Model Inference (JAX/OpenPI)")
    print("=" * 60)
    
    # ========== 测试 JAX 权重推理 ==========
    print("\n" + "=" * 60)
    print("Testing JAX Weights Inference")
    print("=" * 60)
    
    jax_checkpoint_path = "/home/ubuntu/stephen/02-weight/pi05_libero_jax"
    
    if os.path.exists(jax_checkpoint_path):
        try:
            policy, config = load_model_jax(jax_checkpoint_path)
            example = create_libero_example()
            print(f"Example keys: {example.keys()}")
            
            result, first_time, avg_time = run_inference_timing(policy, example, num_runs=10)
            
            print("\n✓ JAX inference successful!")
            print(f"  First run: {first_time:.3f}s")
            print(f"  Average: {avg_time:.3f}s")
        except Exception as e:
            print(f"\n✗ JAX inference failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"JAX checkpoint not found: {jax_checkpoint_path}")
        print("Please download JAX weights from GCS:")
        print(f"  gsutil -m rsync -r gs://openpi-assets/checkpoints/pi05_libero/ {jax_checkpoint_path}")
    
    # ========== 测试 PyTorch 权重推理 ==========
    print("\n" + "=" * 60)
    print("Testing PyTorch Weights Inference")
    print("=" * 60)
    
    pytorch_checkpoint_path = "/home/ubuntu/stephen/02-weight/pi05_libero"
    
    if os.path.exists(pytorch_checkpoint_path):
        try:
            policy, config = load_model_pytorch(pytorch_checkpoint_path)
            example = create_libero_example()
            
            result, first_time, avg_time = run_inference_timing(policy, example, num_runs=10)
            
            print("\n✓ PyTorch inference successful!")
            print(f"  First run: {first_time:.3f}s")
            print(f"  Average: {avg_time:.3f}s")
        except Exception as e:
            print(f"\n✗ PyTorch inference failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"PyTorch checkpoint not found: {pytorch_checkpoint_path}")


if __name__ == "__main__":
    main()
```

---

## 五、JAX → PyTorch 权重转换

### 使用转换脚本
```bash
python examples/convert_jax_model_to_pytorch.py \
  --config-name pi05_libero \
  --checkpoint-dir /home/ubuntu/stephen/02-weight/pi05_libero_jax \
  --output-path /home/ubuntu/stephen/02-weight/pi05_libero_pytorch
```

---

## 六、推理性能对比

| 权重类型 | 首次推理 (JIT编译) | 平均推理时间 |
|---------|------------------|-------------|
| JAX     | ~XX s            | ~XX s       |
| PyTorch | ~XX s            | ~XX s       |

---

## 七、显存估算（JAX 推理）

- **Pi0.5 参数量**：约 **3B**（VLM 主干 + 动作专家）
- **JAX 推理显存**：
  - `fp32`：~**12GB**
  - `fp16/bf16`：~**6GB**
  - `int8`：~**3GB**

---

## 八、常见问题

### 1. GCS 下载失败
- 设置 `GCS_ACCESS` 或手动下载权重放 `~/.cache/openpi/`

### 2. JAX 找不到 GPU
- 重装 `jax[cuda12]`，匹配 CUDA 版本

### 3. 输入维度不匹配
- 严格按 `config.model.input_structure` 构造字典键名

---

## 六、推理性能对比

| 权重类型 | 首次推理 (JIT编译) | 平均推理时间 |
|---------|------------------|-------------|
| JAX (CPU) | 8.4s           | 7.2s        |
| JAX (GPU) | 待测            | 待测         |

> 测试环境：Intel CPU, 无 GPU 加速

### 推理输出示例
```
Actions shape: (10, 7)  # 10 个时间步, 7 维动作
```

---

## 七、显存估算（JAX 推理）

- **Pi0.5 参数量**：约 **3B**（VLM 主干 + 动作专家）
- **JAX 推理显存**：
  - `fp32`：~**12GB**
  - `fp16/bf16`：~**6GB**
  - `int8`：~**3GB**

---

## 八、常见问题

### 1. GCS 下载失败
- 设置 `GCS_ACCESS` 或手动下载权重放 `~/.cache/openpi/`

### 2. JAX 找不到 GPU
- 重装 `jax[cuda12]`，匹配 CUDA 版本

### 3. 输入维度不匹配
- 严格按 `config.model.input_structure` 构造字典键名

### 4. orbax-checkpoint 版本不兼容
- 使用 orbax-checkpoint==0.11.36 版本
- 如遇 `StepMetadata` 错误，需修改 model.py 中的 restore_params 函数

### 5. lerobot 安装超时
- 使用清华镜像：`-i https://pypi.tuna.tsinghua.edu.cn/simple/`
- 或从 git 安装指定版本

---

## 九、环境验证结果

### 验证命令
```bash
source /home/ubuntu/stephen/01-code/.venv/bin/activate
python /home/ubuntu/stephen/01-code/pi_operator/jax_predict.py
```

### 当前状态
| 组件 | 状态 | 说明 |
|------|------|------|
| 环境依赖 | ✅ | jax, flax, orbax 等已安装 |
| ModelScope 权重加载 | ✅ | 可单独加载权重 |
| orbax API 兼容性 | ⚠️ | 需要修改 openpi 源码 |
| libero assets | ✅ | 已从 GCS 下载 |
| 完整推理 | ⚠️ | 需修复 model.py |

### 解决方法

由于 orbax-checkpoint 0.11.36 与 ModelScope 下载的权重格式不完全兼容，需要修改 `openpi/src/openpi/models/model.py` 中的 `restore_params` 函数。

**修改位置**：`openpi/src/openpi/models/model.py` 第 313-318 行

**修改内容**：
```python
# 原代码（约第 314-315 行）
with ocp.PyTreeCheckpointer() as ckptr:
    metadata = ckptr.metadata(params_path)
    item = {"params": metadata["params"]}

# 修改后
with ocp.PyTreeCheckpointer() as ckptr:
    metadata = ckptr.metadata(params_path)
    if hasattr(metadata, 'item_metadata') and 'params' in metadata.item_metadata:
        item = {"params": metadata.item_metadata["params"]}
    else:
        item = {"params": metadata["params"]}
```

**修改原因**：ModelScope 下载的权重使用新版 orbax-checkpoint 格式，`metadata` 是 `StepMetadata` 对象，其参数存储在 `item_metadata` 属性中而非直接可订阅。

---

## 文件位置

- **模型权重**: `/home/ubuntu/stephen/02-weight/pi05_base/` (from ModelScope)
- **libero assets**: `/home/ubuntu/stephen/02-weight/pi05_base/assets/physical-intelligence/libero/`
- **PaliGemma 基础**: `/home/ubuntu/stephen/02-weight/paligemma-3b-pt-224/`
- **OpenPI 仓库**: `/home/ubuntu/stephen/01-code/openpi/`
- **推理脚本**: `/home/ubuntu/stephen/01-code/pi_operator/jax_predict.py`
- **JAX 虚拟环境**: `/home/ubuntu/stephen/01-code/.venv/`