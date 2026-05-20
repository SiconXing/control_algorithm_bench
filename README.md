# Function-Fitting RL Benchmark

用强化学习（RL）拟合复杂分段函数的基准测试平台。环境由 C++ 实现以保证速度，通过 pybind11 暴露给 Python，使用 PyTorch 训练 DQN agent。

## 原理

传统的 RL 环境（如 GridWorld）reward 由固定的规则决定。本项目将 reward 信号替换为**自定义分段函数**：

- **State**：x ∈ [0, 1] 上的一个随机采样点
- **Action**：离散 y 值 bin（默认 100 个），均匀覆盖函数值域
- **Reward**：`-|f(x) - y_pred|`（负绝对误差）
- **Episode**：200 步，每步随机采样新的 x

Agent 的目标是最大化累计 reward，等价于学习函数 f(x) 的形状。通过比较不同函数上的拟合效果（MAE），可以快速评估 RL 算法的表达能力。

## 项目结构

```
├── CMakeLists.txt              # C++ 编译配置
├── main.py                     # 入口：单函数演示
├── experiments.py              # 批量实验运行器
├── train.py                    # 训练/评估/绘图工具
├── pyproject.toml              # Python 项目配置
├── requirements.txt            # Python 依赖
├── src/
│   ├── function_env.h          # C++ 环境头文件
│   ├── function_env.cpp        # C++ 环境实现（10 个分段函数）
│   ├── bindings.cpp            # pybind11 Python 绑定
│   ├── agent_factory.py        # DQN agent 工厂
│   └── piecewise_functions.py  # 函数注册表 + Python ground truth
├── build/                      # CMake 构建目录（gitignored）
├── models/                     # 训练好的模型（gitignored）
└── results/                    # 实验结果图表（gitignored）
```

## 环境搭建

```bash
# 1. 克隆仓库
git clone https://github.com/SiconXing/control_algorithm_bench.git
cd control_algorithm_bench

# 2. 安装 Python 依赖（需要 Python ≥ 3.9）
uv sync
# 或者用 pip：pip install -r requirements.txt pybind11

# 3. 编译 C++ 扩展
mkdir -p build && cd build
cmake ..
make -j4
cd ..
```

编译完成后根目录会生成 `function_env.cpython-*.so`，Python 可直接 `import function_env`。

## 使用方法

### 单函数演示

```bash
# 训练默认函数（step_single，800 episodes）
uv run python main.py

# 指定函数和训练轮数
uv run python main.py --func sawtooth --episodes 500
```

可选函数名：
```
step_single    step_multi     sawtooth     staircase    gaussian_bumps
polynomial_cubic   sinusoid_composite   v_shaped_absolute
discontinuous_jump   complex_multi_segment
```

### 批量实验

```bash
# 完整实验（所有 10 个函数，各 800 episodes）
uv run python experiments.py

# 快速验证模式（各 300 episodes）
uv run python experiments.py --quick

# 只测试单个函数
uv run python experiments.py --func staircase
```

### 输出结果

每次运行后在 `results/` 目录生成：

| 文件 | 内容 |
|------|------|
| `training_progress.png` | 训练过程的 reward 和 loss 曲线 |
| `{函数名}.png` | 真实函数曲线 vs agent 预测 |
| `summary.png` | 所有函数对比汇总图（仅批量实验） |

终端会打印汇总表格：

```
Function                  Difficulty      MAE   BestReward  Conv@Ep  Time(s)
--------------------------------------------------------------------------------
polynomial_cubic          easy         0.0054       -19.97      301     17.1
staircase                 medium       0.0083       -69.19      301     17.3
...
```

## 内置分段函数

| 名称 | 难度 | 描述 |
|------|------|------|
| `step_single` | 简单 | 单阶跃：x < 0.5 → -1, x ≥ 0.5 → 1 |
| `step_multi` | 中等 | 3 阶跃：x < 0.3 → -1, 0.3~0.5 → 0, 0.5~0.7 → 0.5, else → 1 |
| `sawtooth` | 中等 | 锯齿波，周期 0.25 |
| `staircase` | 中等 | 5 级递增台阶 |
| `gaussian_bumps` | 中等 | 2 个高斯峰 + 1 个谷 |
| `polynomial_cubic` | 简单 | 三次多项式 2(x-0.5)³ - (x-0.5) |
| `sinusoid_composite` | 中等 | sin(6πx)·0.6 + cos(2πx)·0.4 |
| `v_shaped_absolute` | 中等 | \|2x-1\|·1.5 - 0.5 + 0.15·sin(10πx) |
| `discontinuous_jump` | 困难 | 左侧正弦 + 右侧余弦 + 中间跳变 |
| `complex_multi_segment` | 困难 | 6 段混合（sin / 二次 / 线性 / exp / cos / 三次） |

## 添加新函数

在 [src/function_env.cpp](src/function_env.cpp) 的 `FunctionId` 枚举和 `evaluate_function()` switch 中添加新函数，同时在 [src/piecewise_functions.py](src/piecewise_functions.py) 中注册对应的 Python 实现和元数据，重新编译即可。

## 自定义训练参数

在代码中创建 agent 时传入超参数：

```python
from src.agent_factory import DQNAgent
import function_env

env = function_env.FunctionFittingEnv(function_env.FunctionId.STEP_SINGLE)
agent = DQNAgent(
    state_size=env.get_state_size(),
    action_size=env.get_action_size(),
    hidden_layers=[128, 64],   # 更大网络
    lr=0.0005,                 # 学习率
    gamma=0.99,                # 折扣因子
    batch_size=128,            # batch 大小
    epsilon_decay=0.995,       # 更慢的探索衰减
)
```

## 依赖

- Python ≥ 3.9
- PyTorch ≥ 2.0
- NumPy
- Matplotlib（绘图用，可选）
- pybind11（C++ 绑定）
- CMake ≥ 3.14
- C++17 编译器
