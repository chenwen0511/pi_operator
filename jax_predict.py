#!/usr/bin/env python3
"""
π0.5 模型推理脚本 (JAX/OpenPI 框架)
"""

import sys
import os

# 设置PYTHONPATH以便导入openpi
OPENPI_SRC = "/home/ubuntu/stephen/01-code/openpi/src"
OPENPI_CLIENT_SRC = "/home/ubuntu/stephen/01-code/openpi/packages/openpi-client/src"
if OPENPI_SRC not in sys.path:
    sys.path.insert(0, OPENPI_SRC)
if OPENPI_CLIENT_SRC not in sys.path:
    sys.path.insert(0, OPENPI_CLIENT_SRC)
os.environ["PYTHONPATH"] = f"{OPENPI_SRC}:{OPENPI_CLIENT_SRC}"

import jax
import numpy as np
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.policies import libero_policy


def load_model(checkpoint_path="/home/ubuntu/stephen/02-weight/pi05_libero"):
    """加载 JAX OpenPI 模型"""
    print(f"Loading model from: {checkpoint_path}")
    print(f"JAX devices: {jax.devices()}")

    # 使用pi05_libero配置
    config = _config.get_config("pi05_libero")
    print(f"Config: {config.name}")
    print(f"Model: {config.model}")

    # 创建训练好的策略
    policy = policy_config.create_trained_policy(config, checkpoint_path)
    print("Model loaded successfully!")

    return policy, config


def create_dummy_example():
    """创建虚拟输入示例"""
    return libero_policy.make_libero_example()


def run_inference(policy, example):
    """运行推理"""
    print("Running inference...")

    result = policy.infer(example)
    print(f"Result keys: {result.keys()}")
    print(f"Actions shape: {result['actions'].shape}")
    print(f"Actions: {result['actions']}")

    return result


def main():
    print("=" * 50)
    print("π0.5 VLA Model Inference (JAX/OpenPI)")
    print("=" * 50)

    # 1. 加载模型
    model_path = "/home/ubuntu/stephen/02-weight/pi05_libero"
    policy, config = load_model(model_path)

    # 2. 准备输入（使用虚拟数据测试）
    print("\nPreparing input example...")
    example = create_dummy_example()
    print(f"Example keys: {example.keys()}")

    # 3. 运行推理
    print("\nRunning inference...")
    try:
        result = run_inference(policy, example)
        print(f"\n✓ Inference successful!")
    except Exception as e:
        print(f"\n✗ Inference failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
