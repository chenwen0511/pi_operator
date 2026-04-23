# PyTorch Profiler 结果汇总

生成时间: 2026-04-23T19:23:00.951802

## 分析结果

### builtin_modules

- 设备: cpu
- 总算子数: 10

| 排名 | 算子 | CPU时间(μs) | CUDA时间(μs) | 调用次数 |
|------|------|------------|------------|----------|
| 1 | `aten::addmm` | 37.83ms | 0.00ms | 200 |
| 2 | `aten::copy_` | 5.15ms | 0.00ms | 200 |
| 3 | `aten::gelu` | 3.83ms | 0.00ms | 100 |
| 4 | `aten::linear` | 0.20ms | 0.00ms | 200 |
| 5 | `aten::transpose` | 0.16ms | 0.00ms | 200 |
| 6 | `aten::t` | 0.15ms | 0.00ms | 200 |
| 7 | `aten::as_strided` | 0.13ms | 0.00ms | 400 |
| 8 | `aten::expand` | 0.13ms | 0.00ms | 200 |
| 9 | `aten::resolve_conj` | 0.02ms | 0.00ms | 400 |
| 10 | `[memory]` | 0.00ms | 0.00ms | 300 |

### attention_layer

- 设备: cpu
- 总算子数: 24

| 排名 | 算子 | CPU时间(μs) | CUDA时间(μs) | 调用次数 |
|------|------|------------|------------|----------|
| 1 | `aten::addmm` | 127.88ms | 0.00ms | 100 |
| 2 | `aten::copy_` | 12.83ms | 0.00ms | 350 |
| 3 | `aten::bmm` | 5.15ms | 0.00ms | 100 |
| 4 | `aten::_softmax` | 1.11ms | 0.00ms | 50 |
| 5 | `aten::mul` | 0.83ms | 0.00ms | 50 |
| 6 | `aten::linear` | 0.37ms | 0.00ms | 100 |
| 7 | `aten::matmul` | 0.33ms | 0.00ms | 100 |
| 8 | `aten::reshape` | 0.30ms | 0.00ms | 300 |
| 9 | `aten::expand` | 0.26ms | 0.00ms | 300 |
| 10 | `aten::view` | 0.22ms | 0.00ms | 300 |

