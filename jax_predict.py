#!/usr/bin/env python3
"""
pi0.5 Model Inference Script (JAX/OpenPI Framework)
Using local ModelScope weights with downloaded libero assets
"""

import sys
import os
import time
import dataclasses

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


def load_model(checkpoint_path):
    """Load JAX OpenPI model"""
    print(f"Loading from: {checkpoint_path}")
    print(f"JAX devices: {jax.devices()}")

    config = _config.get_config("pi05_libero")
    print(f"Config: {config.name}")
    print(f"Model: {config.model}")

    policy = policy_config.create_trained_policy(config, checkpoint_path)
    print("Model loaded!")
    return policy, config


def create_example():
    return libero_policy.make_libero_example()


def run_inference(policy, example, num_runs=10):
    print(f"\nRunning {num_runs} inferences...")

    print("First (with JIT)...")
    start = time.time()
    result = policy.infer(example)
    first_time = time.time() - start
    print(f"  First: {first_time:.3f}s, actions: {result['actions'].shape}")

    times = []
    for _ in range(num_runs):
        start = time.time()
        result = policy.infer(example)
        times.append(time.time() - start)

    avg = sum(times) / len(times)
    print(f"  Avg: {avg:.3f}s, min: {min(times):.3f}s, max: {max(times):.3f}s")
    return result, first_time, avg


def main():
    print("=" * 60)
    print("pi0.5 VLA Model Inference")
    print("Using ModelScope weights + libero assets")
    print("=" * 60)

    checkpoint = "/home/ubuntu/stephen/02-weight/pi05_base"

    if os.path.exists(checkpoint):
        try:
            policy, config = load_model(checkpoint)
            example = create_example()
            print(f"Example keys: {example.keys()}")

            result, first_time, avg_time = run_inference(policy, example, num_runs=10)

            print("\n" + "=" * 60)
            print("SUCCESS!")
            print(f"  First run: {first_time:.3f}s")
            print(f"  Average: {avg_time:.3f}s")
            print("=" * 60)

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Not found: {checkpoint}")


if __name__ == "__main__":
    main()