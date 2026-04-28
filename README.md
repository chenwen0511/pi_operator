# 模型算子与结构分析项目

本项目用于对不同模型进行**算子级**与**模型结构级**分析拆解，目标是把各模型的核心组件、推理流程与实现细节梳理清楚，形成可复用的分析与验证方法。

## 项目定位

- 面向多模型：同一仓库内对多个模型进行并行分析
- 面向算子：关注关键算子的功能、输入输出与实现路径
- 面向结构：梳理模型主干结构、模块关系与执行流程
- 面向落地：保留可运行脚本，便于复现、验证与对比

## 目录组织

项目已按不同模型拆分目录，每个目录独立维护对应模型的分析与实验内容：

- `pi_jax/`：PI 系列（JAX 方向）相关分析与代码
- `pi_torch/`：PI 系列（PyTorch 方向）相关分析与代码
- `qwen/`：Qwen 系列模型算子与结构分析
- `xvla/`：XVLA 模型算子与结构分析

## 本工程目录结构说明

```text
pi_operator/
├── README.md                 # 项目总览与导航
├── torchinfo_analysis.py     # 通用模型结构统计/分析脚本
├── pi_jax/                   # PI(JAX) 模型分析目录
│   ├── jax_pi_predict.md
│   └── jax_predict.py
├── pi_torch/                 # PI(PyTorch) 模型分析目录
│   ├── README_PI_TORCH.md
│   ├── PI_Operator_report.md
│   ├── PI_Operator_report.pdf
│   ├── pi05_structure.txt
│   └── predict.py
├── qwen/                     # Qwen 模型分析目录
│   ├── Qwen_VL_4B_Operator.md
│   ├── Qwen_VL_4B_Operator.pdf
│   └── qwen_predict.py
└── xvla/                     # XVLA 模型分析目录
    ├── XVLA_Opeator.md
    ├── XVLA_Opeator.pdf
    └── xvla_predict.py
```

说明：以上为当前仓库的实际文件结构，后续新增文件可按对应模型目录继续扩展。

## 推荐目录规范

为便于后续统一维护，建议各模型目录逐步保持类似结构：

- `README.md` 或模型说明文档：该模型的背景、结论与使用方式
- `*_report.md`：算子/结构拆解报告
- `predict.py`、`analysis.py` 等脚本：推理或分析验证代码
- 其他辅助文件：配置、结果记录、对比实验脚本等

## 使用方式

1. 进入目标模型目录（如 `pi_torch/`、`qwen/`）
2. 阅读该目录文档，了解模型与分析重点
3. 按需运行对应脚本进行推理或结构验证
4. 将结论沉淀到该目录文档，保持模型间可横向对比

## 当前状态

- 已按模型完成目录拆分
- 各模型可在各自目录内独立迭代
- 后续可持续补充统一模板（算子表、结构图、性能对比项）
