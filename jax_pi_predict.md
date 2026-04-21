# pi0.5 JAX 推理指南

## 快速开始

```bash
# 激活环境
source /home/ubuntu/stephen/01-code/.venv/bin/activate

# 运行推理
python /home/ubuntu/stephen/01-code/pi_operator/jax_predict.py
```

## 依赖安装

```bash
pip install jax==0.10.0 jaxlib==0.10.0
pip install flax==0.12.6 einops transformers huggingface_hub gcsfs
pip install orbax-checkpoint==0.11.36
pip install jaxtyping==0.2.36
pip install chex tqdm_loggable numpydantic dm-tree augmax imageio opencv-python
pip install tyro pytest sentencepiece beartype typer rich polars
pip install git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
```

## 下载模型权重

```bash
# ModelScope 下载
pip install modelscope
modelscope download --model hairuoliu/pi05_base --local_dir /home/ubuntu/stephen/02-weight/pi05_base
```

## 下载 libero assets

```bash
# 从 GCS 下载 norm_stats.json
gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero /home/ubuntu/stephen/02-weight/pi05_base/assets/physical-intelligence/
```

## orbax 兼容性问题修复

修改 `openpi/src/openpi/models/model.py` 第 314-318 行：

```python
# 原代码
metadata = ckptr.metadata(params_path)
item = {"params": metadata["params"]}

# 修改为
metadata = ckptr.metadata(params_path)
if hasattr(metadata, 'item_metadata') and 'params' in metadata.item_metadata:
    item = {"params": metadata.item_metadata["params"]}
else:
    item = {"params": metadata["params"]}
```

## GPU 推理配置

### 前置条件

GPU 推理需要以下条件：
1. NVIDIA  GPU 卡
2. NVIDIA 驱动（已安装）
3. **CUDA 运行时库**（需额外安装）
4. CUDA 驱动版本 >= 12.1

### 1. 检查 GPU 和 CUDA 环境

```bash
# 查看 GPU
nvidia-smi

# 查看是否有 CUDA 运行时库
ls /usr/local/cuda/lib64/libcudart.so* 2>/dev/null
# 如果没有输出，说明需要安装 CUDA
```

### 2. 安装 CUDA 运行时库（必须）

**方式1: apt 安装（推荐）**
```bash
sudo apt update
sudo apt install cuda-toolkit-12-8
```

**方式2: 下载 CUDA Installer**
```bash
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_565.77_linux.run
sudo ./cuda_12.8.0_565.77_linux.run
# 只选择 "CUDA Toolkit" 即可，不需要 driver
```

### 3. 安装 JAX GPU 版本

```bash
pip install jax[cuda12]==0.10.0 --force-reinstall
```

### 4. 设置环境变量

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
```

### 5. 验证 GPU 可用

```bash
python -c "import jax; print(jax.devices())"
# 期望输出: [cuda(id=0)]
```

### 6. 运行推理

```bash
python /home/ubuntu/stephen/01-code/pi_operator/jax_predict.py
```

## 推理结果

### CPU 模式
```
============================================================
pi0.5 VLA Model Inference
Using ModelScope weights + libero assets
============================================================

Loading from: /home/ubuntu/stephen/02-weight/pi05_base
WARNING:jax._src.xla_bridge:An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
JAX devices: [CpuDevice(id=0)]
Config: pi05_libero
Model: Pi0Config(...)
Model loaded!
Example keys: dict_keys(['observation/state', 'observation/image', 'observation/wrist_image', 'prompt'])

Running 10 inferences...
First (with JIT)...
  First: 8.308s, actions: (10, 7)
  Avg: 7.147s, min: 6.964s, max: 7.249s

============================================================
SUCCESS!
  First run: 8.308s
  Average: 7.147s
============================================================
```

### GPU 模式（预期）
```
JAX devices: [cuda(id=0)]
# 首次推理: ~2-3s
# 平均推理: ~1s
```

## 性能对比

| 模式 | 首次推理 | 平均推理 | 加速比 |
|------|---------|---------|-------|
| CPU | 8.3s | 7.1s | 1x |
| GPU | ~2-3s | ~1s | 5-7x |

## 常见问题

### 问题1: JAX 找不到 GPU

**错误信息:**
```
WARNING: An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
JAX devices: [CpuDevice(id=0)]
```

**排查步骤:**
1. 确认安装 CUDA 版本 JAX: `pip install jax[cuda12]`
2. **确认 CUDA 运行时库已安装**:
   ```bash
   ls /usr/local/cuda/lib64/libcudart.so*
   ```
   如果没有输出，需要安装 CUDA Toolkit
3. 设置环境变量: `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`
4. 验证: `python -c "import jax; print(jax.devices())"`

### 问题2: CUDA 运行时库不存在

**错误信息:**
```
RuntimeError: jaxlib/cuda/versions_helpers.cc:135: operation cuInit(0) failed: CUDA_ERROR_UNKNOWN
```

**解决:**
这是因为系统只有 NVIDIA 驱动，没有 CUDA 运行时库。必须安装 CUDA Toolkit：
```bash
sudo apt install cuda-toolkit-12-8
```

### 问题3: CUDA 版本不匹配

**解决:**
确保 jax[cuda12] 版本与系统 CUDA 驱动兼容

### 问题3: jaxtyping 版本不兼容

**错误信息:**
```
AttributeError: module 'jaxtyping._decorator' has no attribute '_check_dataclass_annotations'
```

**解决:**
```bash
pip install jaxtyping==0.2.36
```

## 文件位置

| 类型 | 路径 |
|------|------|
| 模型权重 | `/home/ubuntu/stephen/02-weight/pi05_base/` |
| libero assets | `/home/ubuntu/stephen/02-weight/pi05_base/assets/physical-intelligence/libero/` |
| OpenPI | `/home/ubuntu/stephen/01-code/openpi/` |
| 推理脚本 | `/home/ubuntu/stephen/01-code/pi_operator/jax_predict.py` |
| JAX 环境 | `/home/ubuntu/stephen/01-code/.venv/` |
