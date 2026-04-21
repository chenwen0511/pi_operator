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

## 推理结果

```
============================================================
pi0.5 VLA Model Inference
Using ModelScope weights + libero assets
============================================================

Loading from: /home/ubuntu/stephen/02-weight/pi05_base
JAX devices: [CpuDevice(id=0)]
Config: pi05_libero
Model: Pi0Config(...)
Model loaded!
Example keys: dict_keys(['observation/state', 'observation/image', 'observation/wrist_image', 'prompt'])

Running 10 inferences...
First (with JIT)...
  First: 8.415s, actions: (10, 7)
  Avg: 7.234s, min: 6.911s, max: 7.410s

============================================================
SUCCESS!
  First run: 8.415s
  Average: 7.234s
============================================================
```

## 文件位置

| 类型 | 路径 |
|------|------|
| 模型权重 | `/home/ubuntu/stephen/02-weight/pi05_base/` |
| libero assets | `/home/ubuntu/stephen/02-weight/pi05_base/assets/physical-intelligence/libero/` |
| OpenPI | `/home/ubuntu/stephen/01-code/openpi/` |
| 推理脚本 | `/home/ubuntu/stephen/01-code/pi_operator/jax_predict.py` |
| JAX 环境 | `/home/ubuntu/stephen/01-code/.venv/` |
