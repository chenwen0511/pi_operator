#!/usr/bin/env python3
"""
π0.5 模型推理脚本
使用 LeRobot 框架加载和运行推理
"""

import torch
from lerobot.policies.pi05 import PI05Policy
from lerobot.policies.factory import make_pre_post_processors


def load_model(model_path="/home/ubuntu/stephen/02-weight/pi05_libero"):
    """加载模型"""
    print(f"Loading model from: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    policy = PI05Policy.from_pretrained(model_path).to(device).eval()
    print("Model loaded successfully!")

    # 使用本地 paligemma tokenizer
    tokenizer_path = "/home/ubuntu/stephen/02-weight/paligemma-3b-pt-224"

    # 创建预处理器
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        model_path,
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
            "tokenizer_processor": {"tokenizer_name": tokenizer_path},
        },
    )

    return policy, preprocess, postprocess, device


def create_dummy_example():
    """创建虚拟输入示例（用于测试）"""
    import numpy as np

    # 根据 config.json: 需要两个相机图像 (3, 256, 256) 和 state (8维)
    dummy_image = torch.randn(3, 256, 256)
    dummy_image2 = torch.randn(3, 256, 256)

    # state (8维)
    dummy_state = torch.randn(8)

    example = {
        "observation.images.image": dummy_image,
        "observation.images.image2": dummy_image2,
        "observation.state": dummy_state,
        "task": "pick up the object",
    }
    return example


def run_inference(policy, preprocess, postprocess, example, device):
    """运行推理"""
    print("Running inference...")

    # 预处理
    batch = preprocess(example)
    print(f"Batch keys: {batch.keys() if batch else 'None'}")

    # 只转换 tensor 类型
    batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}

    # 推理
    with torch.inference_mode():
        pred_action = policy.select_action(batch)
        pred_action = postprocess(pred_action)

    print(f"Action: {pred_action}")
    return pred_action


def main():
    print("=" * 50)
    print("π0.5 VLA Model Inference (LeRobot)")
    print("=" * 50)

    # 1. 加载模型
    model_path = "/home/ubuntu/stephen/02-weight/pi05_libero"
    policy, preprocess, postprocess, device = load_model(model_path)

    # 2. 准备输入（使用虚拟数据测试）
    print("\nPreparing input example...")
    example = create_dummy_example()

    # 3. 运行推理
    print("\nRunning inference...")
    try:
        action = run_inference(policy, preprocess, postprocess, example, device)
        print(f"\n✓ Inference successful!")
        print(f"  Action shape: {action.shape}")
    except Exception as e:
        print(f"\n✗ Inference failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
