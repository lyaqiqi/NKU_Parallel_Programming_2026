# 实验一：矩阵列向量内积

## 目录结构

```
exp1/
├── src/
│   └── exp1.cpp              # C++ 核心实现
├── scripts/
│   └── visualize_exp1.py     # 可视化（2张图）
├── build/                    # 编译产物（自动生成）
├── results/
│   ├── exp1_results.csv      # 测试结果（自动生成）
│   └── figures/              # 图表输出（自动生成）
├── run_exp1.bat              # 一键编译+测试（Windows）
└── README.md
```

## 运行步骤

### 1. 测试

```
run_exp1.bat
```

前提：MinGW 中 `g++` 已在 PATH。脚本自动编译 -O0/-O1/-O2/-O3 四个版本并批量测试，结果写入 `results\exp1_results.csv`。

### 2. 可视化

```
python scripts\visualize_exp1.py
```

生成 2 张图：

| 文件 | 内容 |
|------|------|
| `fig1_time_speedup.png` | 时间曲线 + 加速比（双Y轴，固定-O2） |
| `fig2_compiler_effect.png` | 编译优化等级对两种算法的效果对比 |

## 算法说明

| 算法 | 内存访问模式 | cache 行为 |
|------|------------|-----------|
| 平凡算法 | `b[j*n+i]`，外列内行，步长=n | 每次访问均跨行，cache 不友好 |
| 优化算法 | `b[j*n+i]`，外行内列，步长=1 | 顺序访问，cache 友好 |

## 规模设计

| 阶段 | 范围 | 步进 |
|------|------|------|
| L1 内（≤32KB） | n = 16 → 128 | 16 |
| L1→L2 | n = 144 → 512 | 16 |
| L2→L3 | n = 576 → 1024 | 64 |
| L3 以上 | n = 2048, 4096 | — |

> double 矩阵占用 8n² 字节：L1(32KB)≈n=64，L2(256KB)≈n=180

## CSV 格式

```
opt_label,n,runs,t_naive_ms,t_opt_ms,correct
O2,256,100,45.2300,12.1800,OK
```

- `t_naive_ms` / `t_opt_ms`：`runs` 次总耗时（毫秒）
- 可视化脚本中自动除以 `runs` 得到单次平均值
