#!/usr/bin/env python3
"""
PI05 模型 torchinfo 分析
"""

import sys

sys.path.insert(0, "/home/ubuntu/stephen/01-code/lerobot/src")

import torch
from torchinfo import summary
from lerobot.policies.pi05 import PI05Policy


def main():
    model_path = "/home/ubuntu/stephen/02-weight/pi05_libero"

    print("Loading model...")
    # 强制加载到 CPU
    policy = PI05Policy.from_pretrained(model_path, device_map="cpu")
    policy = policy.cpu()
    model = policy.model

    # Vision Tower
    print("\n" + "=" * 60)
    print("1. Vision Tower (SigLIP)")
    print("=" * 60)
    vision = model.paligemma_with_expert.paligemma.model.vision_tower
    summary(
        vision,
        input_data=torch.randn(1, 3, 256, 256),
        depth=2,
        col_names=["input_size", "output_size", "num_params"],
    )

    # Language Model
    print("\n" + "=" * 60)
    print("2. Language Model (Gemma 2B)")
    print("=" * 60)
    lm = model.paligemma_with_expert.paligemma.model.language_model
    tokens = torch.randint(0, 32000, (1, 50))
    attention_mask = torch.ones(1, 50)
    summary(
        lm,
        input_data=[tokens, attention_mask],
        depth=2,
        col_names=["input_size", "output_size", "num_params"],
    )

    # Action Expert
    print("\n" + "=" * 60)
    print("3. Action Expert (Gemma 300M)")
    print("=" * 60)
    expert = model.paligemma_with_expert.gemma_expert
    expert_tokens = torch.randint(0, 32000, (1, 50))
    expert_mask = torch.ones(1, 50)
    summary(
        expert,
        input_data=[expert_tokens, expert_mask],
        depth=2,
        col_names=["input_size", "output_size", "num_params"],
    )

    # Action Heads
    print("\n" + "=" * 60)
    print("4. Action Heads")
    print("=" * 60)
    print(f"action_in_proj: {model.action_in_proj}")
    print(f"action_out_proj: {model.action_out_proj}")
    print(f"time_mlp_in: {model.time_mlp_in}")
    print(f"time_mlp_out: {model.time_mlp_out}")

    # 测试 action_in_proj
    action_in = torch.randn(1, 50, 8)
    summary(
        model.action_in_proj,
        input_data=action_in,
        depth=0,
        col_names=["input_size", "output_size", "num_params"],
    )

    print("\n" + "=" * 60)
    total = sum(p.numel() for p in model.parameters())
    print("总参数量: {:.2f}B".format(total / 1e9))
    print("=" * 60)


if __name__ == "__main__":
    main()
