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

            result, first_time, avg_time = run_inference_timing(
                policy, example, num_runs=10
            )

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
        print(
            f"  gsutil -m rsync -r gs://openpi-assets/checkpoints/pi05_libero/ {jax_checkpoint_path}"
        )

    # ========== 测试 PyTorch 权重推理 ==========
    print("\n" + "=" * 60)
    print("Testing PyTorch Weights Inference")
    print("=" * 60)

    pytorch_checkpoint_path = "/home/ubuntu/stephen/02-weight/pi05_libero"

    if os.path.exists(pytorch_checkpoint_path):
        try:
            policy, config = load_model_pytorch(pytorch_checkpoint_path)
            example = create_libero_example()

            result, first_time, avg_time = run_inference_timing(
                policy, example, num_runs=10
            )

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
