# π0.5 (PI05) Conda 环境复现

目标：在隔离的 conda 环境中安装依赖并跑通 `predict.py`，避免继续使用物理机全局 Python 环境。

## 已验证路径

- 脚本：`pi_torch/predict.py`
- 模型路径：`/home/ubuntu/stephen/02-weight/pi05_libero_finetuned_v044/pi05_libero_finetuned_quantiles`
- Tokenizer 路径：`/home/ubuntu/stephen/02-weight/paligemma-3b-pt-224`

## 1) 创建并激活 conda 环境

若机器全局配置里优先使用 `conda-forge`（常见于 `~/.condarc`），即便命令里写了 `-c defaults`，仍可能合并多个 channel，`Collecting package metadata (repodata.json)` 会很久。建议用下面任一方式：

**方式 A（推荐）：只读 defaults channel，跳过全局 channel 列表**

```bash
conda create -n pi05 -y --override-channels -c defaults python=3.13 pip
conda activate pi05
```

**方式 B（最快）：不用 conda，纯 venv（你已安装 Python 3.13 时适用）**

```bash
cd /home/ubuntu/stephen/01-code/pi_operator/pi_torch
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python predict.py
```

## 2) 安装依赖

在 `pi_torch` 目录执行：

```bash
cd /home/ubuntu/stephen/01-code/pi_operator/pi_torch
python -m pip install -U pip
python -m pip install -r requirements.txt
```

`requirements.txt` 已固定关键兼容版本：

- `torch==2.10.0`
- `transformers==5.3.0`
- `lerobot==0.5.1`
- `numpy==2.2.6`
- `gymnasium==1.2.3`
- `safetensors==0.7.0`

## 3) 运行推理

```bash
cd /home/ubuntu/stephen/01-code/pi_operator/pi_torch
python predict.py
```

## 4) 成功判定

出现以下日志即表示跑通：

- `✓ Inference successful!`
- `Action shape: torch.Size([1, 7])`

`AUTOTUNE ...` 大段输出是 Triton/Inductor 自动调优日志，属于正常现象。

## 本次定位结论（摘要）

1. 先修复了错误模型路径（原路径不存在）。
2. 随后定位到 `transformers` 版本不兼容导致：
   `AttributeError: 'Tensor' object has no attribute 'pooler_output'`。
3. 将 `transformers` 固定为 `5.3.0` 后推理成功。
