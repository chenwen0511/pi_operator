#!/usr/bin/env python3
"""
PI05 模型可视化 - 打印模型结构
"""

import sys

sys.path.insert(0, "/home/ubuntu/stephen/01-code/lerobot/src")

import torch
from lerobot.policies.pi05 import PI05Policy


def print_model_structure():
    """打印模型结构"""

    model_path = "/home/ubuntu/stephen/02-weight/pi05_libero"

    print("Loading PI05 model...")
    policy = PI05Policy.from_pretrained(model_path, device_map="cpu")

    model = policy.model

    print("\n" + "=" * 60)
    print("π0.5 模型结构")
    print("=" * 60)

    # 打印顶层结构
    print("\n[1] 顶层模块:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {type(module).__name__} ({params / 1e6:.1f}M 参数)")

    # 打印视觉编码器
    print("\n[2] 视觉编码器 (Vision Tower):")
    if hasattr(model, "paligemma_with_expert"):
        vlm = model.paligemma_with_expert
        if hasattr(vlm, "paligemma"):
            vision = vlm.paligemma.model.vision_tower
            print(f"  类型: {type(vision).__name__}")
            for name, module in list(vision.named_children())[:5]:
                params = (
                    sum(p.numel() for p in module.parameters())
                    if hasattr(module, "parameters")
                    else 0
                )
                print(f"    {name}: {type(module).__name__}")

    # 打印语言模型
    print("\n[3] 语言模型 (Language Model):")
    if hasattr(model, "paligemma_with_expert"):
        vlm = model.paligemma_with_expert
        if hasattr(vlm, "paligemma"):
            lm = vlm.paligemma.model.language_model
            print(f"  类型: {type(lm).__name__}")
            if hasattr(lm, "config"):
                print(f"  Hidden size: {lm.config.hidden_size}")
                print(f"  Num layers: {lm.config.num_hidden_layers}")
                print(f"  Num attention heads: {lm.config.num_attention_heads}")

    # 打印动作 Expert
    print("\n[4] 动作 Expert (Action Expert):")
    if hasattr(model, "paligemma_with_expert"):
        expert = model.paligemma_with_expert.gemma_expert
        print(f"  类型: {type(expert).__name__}")
        if hasattr(expert, "model"):
            exp_model = expert.model
            print(f"  Hidden size: {exp_model.config.hidden_size}")
            print(f"  Num layers: {exp_model.config.num_hidden_layers}")

    # 打印动作头
    print("\n[5] 动作头 (Action Head):")
    if hasattr(model, "action_in_proj"):
        print(f"  action_in_proj: {model.action_in_proj}")
    if hasattr(model, "action_out_proj"):
        print(f"  action_out_proj: {model.action_out_proj}")
    if hasattr(model, "time_mlp_in"):
        print(f"  time_mlp_in: {model.time_mlp_in}")
    if hasattr(model, "time_mlp_out"):
        print(f"  time_mlp_out: {model.time_mlp_out}")

    # 打印总参数量
    print("\n" + "=" * 60)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params / 1e9:.2f}B")
    print("=" * 60)

    # 保存到文件
    output_file = "pi05_structure.txt"
    with open(output_file, "w") as f:
        f.write("π0.5 模型结构\n")
        f.write("=" * 60 + "\n\n")

        def print_tree(module, name="model", indent=0, file=f):
            prefix = "  " * indent
            params = (
                sum(p.numel() for p in module.parameters())
                if hasattr(module, "parameters")
                else 0
            )
            file.write(
                f"{prefix}{name}: {type(module).__name__} ({params / 1e6:.1f}M)\n"
            )
            for child_name, child_module in module.named_children():
                print_tree(child_module, child_name, indent + 1, file)

        print_tree(model, "pi05", file=f)

        f.write(f"\n总参数量: {total_params / 1e9:.2f}B\n")

    print(f"\n模型结构已保存到: {output_file}")

    # 尝试导出较小模块
    print("\n尝试导出子模块...")

    try:
        # 导出 action_in_proj (很小的模块)
        print("  导出 action_in_proj...")
        small_module = model.action_in_proj
        output_path = "pi05_action_in_proj.onnx"

        dummy = torch.randn(1, 50, 8)
        torch.onnx.export(small_module, (dummy,), output_path, verbose=False)

        import os

        size = os.path.getsize(output_path) / 1024
        print(f"  ✓ {output_path} ({size:.1f} KB)")

    except Exception as e:
        print(f"  ✗ 导出失败: {e}")


if __name__ == "__main__":
    print_model_structure()
