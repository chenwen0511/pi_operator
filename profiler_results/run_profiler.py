"""
PyTorch Profiler 实际运行脚本
用于分析 π0.5, XVLA, Qwen3-VL 模型的性能
"""

import torch
import torch.profiler as profiler
import json
import os
import sys
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path(__file__).parent / "profiler_results"
OUTPUT_DIR.mkdir(exist_ok=True)

def get_device():
    if torch.cuda.is_available():
        try:
            return torch.device("cuda")
        except:
            return torch.device("cpu")
    return torch.device("cpu")

def profile_builtin_module():
    """Profile PyTorch 内置模块"""
    print("\n" + "="*60)
    print("Profiling PyTorch Builtin Modules")
    print("="*60)
    
    device = get_device()
    print(f"Device: {device}")
    
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(1024, 2048).to(device)
            self.linear2 = torch.nn.Linear(2048, 1024).to(device)
            self.act = torch.nn.GELU()
        
        def forward(self, x):
            x = self.act(self.linear1(x))
            return self.linear2(x)
    
    model = SimpleModel().eval()
    data = torch.randn(32, 1024, device=device)
    
    activities = [profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(profiler.ProfilerActivity.CUDA)
    
    with profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            for _ in range(100):
                model(data)
    
    key_avg = prof.key_averages()
    print(key_avg.table(sort_by="self_cpu_time_total", row_limit=15))
    
    results = {
        "model": "builtin_modules",
        "device": str(device),
        "timestamp": datetime.now().isoformat(),
        "operators": [
{
                "key": item.key,
                "self_cpu_time_us": item.self_cpu_time_total,
                "cpu_time_us": item.cpu_time_total,
                "self_cuda_time_us": getattr(item, 'self_cuda_time_total', 0),
                "cuda_time_us": getattr(item, 'cuda_time_total', 0),
                "count": item.count,
            }
            for item in key_avg
        ]
    }
    
    results["operators"].sort(key=lambda x: x["self_cpu_time_us"], reverse=True)
    
    with open(OUTPUT_DIR / "builtin_profiler.json", "w") as f:
        json.dump(results, f, indent=2)
    
    if hasattr(prof, 'export_chrome_trace'):
        try:
            prof.export_chrome_trace(str(OUTPUT_DIR / "builtin_trace.json"))
        except:
            pass
    print(f"Results saved to {OUTPUT_DIR / 'builtin_profiler.json'}")
    
    return results


def profile_custom_forward():
    """Profile 自定义前向传播"""
    print("\n" + "="*60)
    print("Profiling Custom Forward Pass")
    print("="*60)
    
    device = get_device()
    
    class AttentionLayer(torch.nn.Module):
        def __init__(self, dim, num_heads=8):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            self.qkv = torch.nn.Linear(dim, dim * 3)
            self.proj = torch.nn.Linear(dim, dim)
        
        def forward(self, x):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            attn = attn.softmax(dim=-1)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            return x
    
    model = AttentionLayer(1024, 8).to(device).eval()
    data = torch.randn(4, 64, 1024, device=device)
    
    activities = [profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(profiler.ProfilerActivity.CUDA)
    
    with profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            for _ in range(50):
                model(data)
    
    key_avg = prof.key_averages()
    print(key_avg.table(sort_by="self_cpu_time_total", row_limit=15))
    
    results = {
        "model": "attention_layer",
        "device": str(device),
        "timestamp": datetime.now().isoformat(),
        "operators": [
            {
                "key": item.key,
                "self_cpu_time_us": item.self_cpu_time_total,
                "cpu_time_us": item.cpu_time_total,
                "self_cuda_time_us": getattr(item, 'self_cuda_time_total', 0),
                "cuda_time_us": getattr(item, 'cuda_time_total', 0),
                "count": item.count,
            }
            for item in key_avg
        ]
    }
    
    results["operators"].sort(key=lambda x: x["self_cpu_time_us"], reverse=True)
    
    with open(OUTPUT_DIR / "attention_profiler.json", "w") as f:
        json.dump(results, f, indent=2)
    
    prof.export_chrome_trace(str(OUTPUT_DIR / "attention_trace.json"))
    
    return results


def profile_full_model(model, batch, name="model"):
    """Profile 完整模型"""
    print("\n" + "="*60)
    print(f"Profiling: {name}")
    print("="*60)
    
    device = get_device()
    model = model.to(device).eval()
    
    activities = [profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(profiler.ProfilerActivity.CUDA)
    
    # Move batch to device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            model(**batch)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Profile
    with profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            for _ in range(10):
                model(**batch)
                if device.type == "cuda":
                    torch.cuda.synchronize()
    
    key_avg = prof.key_averages()
    print(key_avg.table(sort_by="self_cpu_time_total", row_limit=20))
    
    results = {
        "model": name,
        "device": str(device),
        "timestamp": datetime.now().isoformat(),
        "operators": [
            {
                "key": item.key,
                "self_cpu_time_us": item.self_cpu_time_total,
                "cpu_time_us": item.cpu_time_total,
                "self_cuda_time_us": getattr(item, 'self_cuda_time_total', 0),
                "cuda_time_us": getattr(item, 'cuda_time_total', 0),
                "count": item.count,
            }
            for item in key_avg
        ]
    }
    
    results["operators"].sort(key=lambda x: x["self_cpu_time_us"], reverse=True)
    
    with open(OUTPUT_DIR / f"{name}_profiler.json", "w") as f:
        json.dump(results, f, indent=2)
    
    prof.export_chrome_trace(str(OUTPUT_DIR / f"{name}_trace.json"))
    
    return results


def generate_report():
    """生成汇总报告"""
    import glob
    
    json_files = list(OUTPUT_DIR.glob("*_profiler.json"))
    
    if not json_files:
        print("No profiler results found")
        return
    
    with open(OUTPUT_DIR / "summary_report.md", "w") as f:
        f.write("# PyTorch Profiler 结果汇总\n\n")
        f.write(f"生成时间: {datetime.now().isoformat()}\n\n")
        f.write("## 分析结果\n\n")
        
        for json_file in json_files:
            with open(json_file) as jf:
                data = json.load(jf)
            
            model_name = data.get("model", json_file.stem)
            device = data.get("device", "Unknown")
            operators = data.get("operators", [])
            
            f.write(f"### {model_name}\n\n")
            f.write(f"- 设备: {device}\n")
            f.write(f"- 总算子数: {len(operators)}\n\n")
            
            f.write("| 排名 | 算子 | CPU时间(μs) | CUDA时间(μs) | 调用次数 |\n")
            f.write("|------|------|------------|------------|----------|\n")
            
            for i, op in enumerate(operators[:10], 1):
                f.write(f"| {i} | `{op['key']}` | {op['self_cpu_time_us']/1000:.2f}ms | {op['self_cuda_time_us']/1000:.2f}ms | {op['count']} |\n")
            
            f.write("\n")
    
    print(f"Summary report: {OUTPUT_DIR / 'summary_report.md'}")


def main():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    profile_builtin_module()
    profile_custom_forward()
    generate_report()
    
    print("\nDone!")


if __name__ == "__main__":
    main()