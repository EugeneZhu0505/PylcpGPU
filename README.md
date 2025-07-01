# PylcpGPU - GPU加速多原子并行计算库

基于pylcp的GPU加速多原子激光冷却模拟库，支持大规模原子的并行计算。

## 🚀 主要特性

### GPU加速计算
- 使用CuPy实现GPU并行计算
- 支持CUDA 11.x和12.x
- 自动回退到CPU计算（GPU不可用时）
- 显著的性能提升（相比CPU串行计算）

### 大规模并行处理
- 支持批量处理大量原子
- 可配置批量大小优化内存使用
- 内存使用与批量大小成正比，而非总原子数
- 支持超过GPU内存限制的大规模系统

### 完全兼容
- 与原pylcp库完全兼容
- 支持所有物理模型（激光场、磁场、哈密顿量等）
- 无需修改现有代码即可使用
- 保持相同的API接口

## 📦 安装

### 基础依赖
```bash
pip install -r requirements.txt
```

### GPU支持（推荐）
```bash
# 对于CUDA 12.x
pip install cupy-cuda12x

# 对于CUDA 11.x
pip install cupy-cuda11x
```

**注意**: 如果没有GPU或CuPy，代码会自动使用CPU计算。

## 🎯 快速开始

### 基本用法

```python
import pylcp
from pylcp.obe_gpu import obe_gpu
import numpy as np

# 创建GPU加速的OBE求解器
obe_solver = obe_gpu(
    laserBeams=laser_beams,
    magField=mag_field,
    hamitlonian=hamiltonian,
    use_gpu=True,        # 启用GPU加速
    batch_size=1000      # 批量大小
)

# 设置多个原子的初始条件
N_atoms = 5000
rho0_batch = np.zeros((n_states, n_states, N_atoms), dtype=complex)
r0_batch = np.random.randn(3, N_atoms) * 1000
v0_batch = np.random.uniform(-1, 1, (3, N_atoms))

# 设置初始条件
obe_solver.set_initial_conditions_batch(rho0_batch, r0_batch, v0_batch)

# 批量演化
solutions = obe_solver.evolve_motion_batch(
    t_span=[0, 0.02],
    t_eval=np.linspace(0, 0.02, 500)
)

# 分析结果
for i, sol in enumerate(solutions[:10]):
    print(f"Atom {i}: final position = {sol.r[:, -1]}")
```

### 运行演示

```bash
# 完整演示（5000个原子）
python demo_gpu.py

# 性能测试
python test_gpu.py

# 基准测试
python benchmark_gpu.py
```

## 📊 性能对比

| 原子数量 | CPU时间 | GPU时间 | 加速比 |
|---------|---------|---------|--------|
| 100     |         |         |        |
| 1,000   |         |         |        |
| 10,000  |         |         |        |
| 100,000 |         |         |        |

*性能测试数据待补充

## 🔧 API参考

### obe_gpu类

#### 构造函数
```python
obe_gpu(laserBeams, magField, hamitlonian, 
        use_gpu=True, batch_size=1000, **kwargs)
```

**参数**:
- `use_gpu` (bool): 是否使用GPU加速
- `batch_size` (int): 批量处理的原子数量
- 其他参数与原`obe`类相同

#### 主要方法

##### set_initial_conditions_batch
```python
set_initial_conditions_batch(rho0_batch, r0_batch, v0_batch)
```
设置多个原子的初始条件。

##### evolve_motion_batch
```python
evolve_motion_batch(t_span, **kwargs)
```
批量演化多个原子的运动。

##### observable_batch
```python
observable_batch(O, rho_batch=None)
```
计算多个原子的可观测量。

## 💡 使用建议

### 批量大小选择
- **小系统**: batch_size = 较小值
- **中等系统**: batch_size = 中等值
- **大系统**: batch_size = 较大值

*具体数值建议待测试后补充

### 性能优化
- 确保CUDA驱动程序是最新的
- 根据GPU内存调整批量大小
- 对于超大系统，考虑分批处理
- 使用`transform_into_re_im=True`减少内存使用

## 🧪 测试和验证

### 运行测试
```bash
# 正确性测试
python test_gpu.py

# 性能基准测试
python benchmark_gpu.py
```

### 测试内容
- 单原子一致性测试
- 批量处理正确性验证
- 性能缩放测试
- 内存使用测试

## 📁 项目结构

```
PylcpGPU/
├── pylcp/
│   ├── obe.py          # 原始OBE实现
│   ├── obe_gpu.py      # GPU加速实现
│   └── ...
├── demo.py             # 原始演示
├── demo_gpu.py         # GPU演示
├── test_gpu.py         # 测试脚本
├── benchmark_gpu.py    # 基准测试
├── config.py           # 配置文件
├── README.md           # 本文件
├── README_GPU.md       # 详细GPU文档
└── requirements.txt    # 依赖列表
```

## 🔬 技术细节

### 实现原理
1. **矩阵批量化**: 将单原子演化扩展为多原子批量矩阵运算
2. **GPU并行**: 使用CuPy实现GPU上的并行计算
3. **内存管理**: 智能的CPU-GPU数据传输
4. **批量处理**: 将大规模系统分解为可管理的批量

### 数学表示
原始单原子演化：
```
dρ/dt = -i[H, ρ] + L[ρ]
```

批量多原子演化：
```
d/dt [ρ₁, ρ₂, ..., ρₙ] = M × [ρ₁, ρ₂, ..., ρₙ]
```

## ❓ 故障排除

### 常见问题

**Q: 提示"CuPy not available"**
```bash
pip install cupy-cuda12x  # 或对应CUDA版本
```

**Q: GPU内存不足**
- 减少`batch_size`参数
- 分批处理原子

**Q: 结果与CPU版本不一致**
- 检查数值精度设置
- 确保使用相同的随机种子

**Q: 性能提升不明显**
- 确保原子数量足够大
- 检查GPU利用率

## 🤝 贡献

欢迎提交问题报告和功能请求！

### 开发指南
1. Fork项目
2. 创建功能分支
3. 提交更改
4. 运行测试
5. 创建Pull Request

## 📄 许可证

本项目遵循与原pylcp相同的许可证。

## 🙏 致谢

- 原pylcp库的开发者
- CuPy团队提供的GPU加速支持
- 所有贡献者和测试用户

---

**注意**: 这是一个研究项目，用于演示GPU加速在原子物理模拟中的应用。如需在生产环境中使用，请进行充分的验证和测试。