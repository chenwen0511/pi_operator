# PyTorch Profiler 使用指南

本指南介绍如何使用 PyTorch 原生 `torch.profiler` 对模型进行性能分析。

---

## 1. 安装依赖

```bash
pip install torch torchvision
# 确保 CUDA 可用
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 2. 基本用法

### 2.1 简单的 Profiler

```python
import torch
import torch.profiler

with profiler.profile() as prof:
    # 在这里执行你的模型推理
    model(input_data)

# 打印结果
print(prof.key_averages())
```

### 2.2 完整的 Profiler 示例

```python
import torch
import torch.profiler as profiler

model = MyModel().cuda().eval()
data = torch.randn(batch_size, input_dim).cuda()

with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    with torch.no_grad():
        model(data)

# 按 CPU 时间排序
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
```

---

## 3. 使用 schedule

对于长训练任务，使用 schedule 避免生成过大 trace 文件：

```python
from torch.profiler import ProfilerActivity, schedule

with profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=profiler.schedule(
        wait=5,      # 前5步不记录
        warmup=2,    # 接下来2步热身
        active=6,     # 记录6步
        repeat=2,     # 重复2次
    ),
    on_trace_ready=lambda prof: save_trace(prof),
    record_shapes=True,
) as prof:
    for step, batch in enumerate(dataloader):
        model(batch)
        prof.step()
```

---

## 4. 输出格式

### 4.1 表格输出

```python
# 基本表格
key_avg = prof.key_averages()
print(key_avg.table())

# 按时间排序
print(key_avg.table(sort_by="self_cpu_time_total", row_limit=20))

# 按 CUDA 时间排序
print(key_avg.table(sort_by="self_cuda_time_total", row_limit=20))
```

### 4.2 Chrome Trace 格式

```python
# 导出为 Chrome trace 格式，可用 Chrome://tracing 查看
prof.export_chrome_trace("trace.json")
```

在 Chrome 浏览器中打开 `chrome://tracing`，点击 "Load" 加载 `trace.json`。

### 4.3 TensorBoard

```python
from torch.profiler.tensorboard_plugin import ProfilerTensorBoardPlugin

# 导出到 TensorBoard
prof.export_tb_summary("./tensorboard_summary")
# 或者
prof.export_chrome_trace("./trace.json")
```

然后运行：
```bash
tensorboard --logdir=./tensorboard_summary
```

### 4.4 JSON 格式

```python
import json

key_avg = prof.key_averages()
results = {
    "operators": [
        {
            "key": item.key,
            "self_cpu_time_total": item.self_cpu_time_total,
            "cpu_time_total": item.cpu_time_total,
            "self_cuda_time_total": item.self_cuda_time_total,
            "cuda_time_total": item.cuda_time_total,
            "count": item.count,
        }
        for item in key_avg
    ]
}

with open("profiler_results.json", "w") as f:
    json.dump(results, f, indent=2, default=lambda x: float(x))
```

---

## 5. 参数详解

| 参数 | 类型 | 说明 |
|------|------|------|
| `activities` | list | 要监控的活动：`CPU`, `CUDA`, `XPU`, `MPS` |
| `schedule` | callable | 记录调度 (wait/warmup/active/repeat) |
| `record_shapes` | bool | 记录算子输入形状 |
| `profile_memory` | bool | 记录内存分配/释放 |
| `with_stack` | bool | 记录 Python 栈追踪 |
| `on_trace_ready` | callable | trace 准备好时的回调 |
| `use_gpu` | bool | 是否使用 GPU (已弃用) |

---

## 6. 排序选项

| 排序键 | 说明 |
|--------|------|
| `self_cpu_time_total` | CPU 自时间 (不含子算子) |
| `cpu_time_total` | CPU 总时间 |
| `self_cuda_time_total` | CUDA 自时间 |
| `cuda_time_total` | CUDA 总时间 |
| `cpu_time_total` | CPU 总时间 |
| `self_cuda_time_total` | CUDA 自时间 |

---

## 7. 实际分析脚本

```python
import torch
import torch.profiler as profiler
import json
from pathlib import Path

OUTPUT_DIR = Path("profiler_results")
OUTPUT_DIR.mkdir(exist_ok=True)


def analyze_model(model, input_data, name="model", num_runs=10):
    device = next(model.parameters()).device
    is_cuda = device.type == "cuda"
    
    activities = [profiler.ProfilerActivity.CPU]
    if is_cuda:
        activities.append(profiler.ProfilerActivity.CUDA)
    
    # 预热
    for _ in range(3):
        with torch.no_grad():
            model(input_data)
    
    # 实际测量
    with profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=is_cuda,
        with_stack=is_cuda,
    ) as prof:
        with torch.no_grad():
            for _ in range(num_runs):
                model(input_data)
    
    return prof


def export_results(prof, name):
    key_avg = prof.key_averages()
    
    results = {
        "model": name,
        "total_runs": 1,
        "operators": [
            {
                "operator": item.key,
                "self_cpu_time_us": item.self_cpu_time_total,
                "cpu_time_us": item.cpu_time_total,
                "self_cuda_time_us": item.self_cuda_time_total,
                "cuda_time_us": item.cuda_time_total,
                "count": item.count,
                "input_shapes": str(getattr(item, 'input_shapes', [])),
            }
            for item in key_avg
        ]
    }
    
    # 排序按 CPU 时间
    results["operators"].sort(key=lambda x: x["self_cpu_time_us"], reverse=True)
    
    # 保存 JSON
    json_file = OUTPUT_DIR / f"{name}_profiler.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x))
    
    # 输出表格
    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"{'='*60}")
    print(key_avg.table(sort_by="self_cpu_time_total", row_limit=20))
    
    # 保存 Chrome trace
    trace_file = OUTPUT_DIR / f"{name}_trace.json"
    prof.export_chrome_trace(str(trace_file))
    print(f"\nTrace saved to: {trace_file}")
    
    return results


if __name__ == "__main__":
    # 示例使用
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1024, 1024)
            self.act = torch.nn.GELU()
        
        def forward(self, x):
            return self.act(self.linear(x))
    
    model = SimpleModel().cuda()
    data = torch.randn(32, 1024).cuda()
    
    print("Profiling model...")
    prof = analyze_model(model, data, "simple_model", num_runs=100)
    export_results(prof, "simple_model")
```

---

## 8. 运行分析

```bash
# 分析你的模型
python analyze_model.py

# 查看结果
cat profiler_results/simple_model_profiler.json

# 在 TensorBoard 查看
tensorboard --logdir=profiler_results

# 在 Chrome 查看 trace
# 打开 chrome://tracing，加载 profiler_results/simple_model_trace.json
```

---

## 9. 结果解读

### 9.1 时间分析

- **self_cpu_time**: 算子本身的时间（不含子调用）
- **cpu_time**: 算子 + 所有子调用的时间
- **cuda_time**: GPU kernel 执行���间
- **count**: 调用次数

### 9.2 内存分析

当 `profile_memory=True` 时：

- **self_cpu_gpu_allocated**: 算子分配的 GPU 内存
- **self_cpu_gpu_reserved**: 算子保留的 GPU 内存
- **self_cpu_gpu_freed**: 算子释放的 GPU 内存

### 9.3 性能瓶颈

关注：
1. 时间占比高的算子
2. 调用次数多的算子
3. 内存分配大的算子

---

## 10. TensorBoard 查看 Profiler

```bash
# 安装 TensorBoard profiler plugin
pip install torch_tb_profiler

# 运行分析
python analyze_model.py

# 查看
tensorboard --logdir=./profiler_results
```

打开 `http://localhost:6006` 查看 PyTorch Profiler 插件。

---

## 11. 常见问题

### 11.1 CUDA 不可用

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
```

### 11.2 Profiler 开销大

- 减少 `record_shapes` 和 `with_stack`
- 使用 `schedule` 限制记录步数
- warmup 预热

### 11.3 Trace 文件过大

使用 schedule 限制：
```python
schedule=profiler.schedule(wait=5, warmup=2, active=6, repeat=1)
```