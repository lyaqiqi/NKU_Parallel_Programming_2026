# ANN (Approximate Nearest Neighbor) Lab - CLAUDE.md

## 项目概述

南开大学并行程序设计 2026，Lab03 —— 近似最近邻（ANN）搜索优化实验。

**目标**：在 DEEP100K 数据集上实现高召回率（recall@10）且低延迟的向量搜索，通过修改 `main.cc` 中的查询逻辑（和/或引入自己的索引/搜索方法）来超越基准暴力搜索 `flat_search`。

---

## 目录结构

```
ann/
├── main.cc          # 核心评测代码（可部分修改，见限制说明）
├── flat_scan.h      # 基准暴力搜索（禁止修改）
├── simd_utils.h     # simd8float32 封装（ARM NEON / x86 AVX / 标量三路）+ simd_inner_product
├── simd_search.h    # simd_flat_search v1–v4：SIMD / FMA / 3路展开 / 预取，消融对比用
├── sq_search.h      # SQIndex：int8 量化索引 + int8×int8 SIMD 两阶段搜索
├── test.sh          # gzip 自解压脚本（测试辅助）
├── qsub.sh          # PBS 集群作业提交脚本
└── hnswlib/         # HNSW 近似最近邻库（第三方，只读使用）
    ├── hnswlib/
    │   ├── hnswlib.h        # 主入口头文件，SIMD 能力检测，接口定义
    │   ├── hnswalg.h        # HierarchicalNSW 核心算法实现
    │   ├── bruteforce.h     # 暴力搜索实现
    │   ├── space_ip.h       # 内积距离空间（含 AVX/AVX512/SSE 优化）
    │   ├── space_l2.h       # L2 距离空间
    │   ├── visited_list_pool.h  # 访问列表池（多线程支持）
    │   └── stop_condition.h
    └── examples/            # 使用示例
```

---

## 核心文件详解

### [main.cc](main.cc)

评测框架，不可大幅改动，**学生只能修改搜索函数的调用方式**（函数名、参数、改为调用成员函数），**不能修改函数返回值类型**。

**主要逻辑**：
1. `LoadData<T>()` —— 读取二进制向量文件（格式：4字节 n + 4字节 d + n×d×sizeof(T) 数据）
2. 加载三个数据集文件（路径 `/anndata/`）：
   - `DEEP100K.query.fbin`：查询向量（float）
   - `DEEP100K.gt.query.100k.top100.bin`：ground truth（int，每条查询 top-100）
   - `DEEP100K.base.100k.fbin`：底库向量（float，100K 条）
3. 只测试前 **2000** 条查询，`k=10`
4. 对每条查询：计时 → 调用搜索函数 → 与 ground truth top-10 比对 → 记录 recall 和延迟（us）
5. 输出平均 recall 和平均延迟

**可选索引构建**（已注释掉）：
```cpp
// build_index(base, base_number, vecdim);
```
`build_index()` 示例使用 HNSW（`efConstruction=150, M=16`），以 OMP 并行添加点，保存到 `files/hnsw.index`。

**`SearchResult` 结构**：
```cpp
struct SearchResult {
    float recall;
    int64_t latency; // 单位 us
};
```

### [flat_scan.h](flat_scan.h)

**禁止修改**。基准暴力搜索实现。

```cpp
std::priority_queue<std::pair<float, uint32_t>>
flat_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k)
```

算法：对底库所有向量计算内积距离 `dis = 1 - dot(base[i], query)`，维护大小为 k 的最大堆返回 top-k。时间复杂度 O(n×d)，无任何并行或 SIMD 优化。

### [simd_utils.h](simd_utils.h)

提供与架构无关的 `simd8float32` 结构体，封装 256 位（8 × float32）宽的 SIMD 运算。编译时根据宏自动选择后端：

| 宏 | 后端 | 平台 |
|----|------|------|
| `__ARM_NEON` / `__aarch64__` | ARM NEON (`float32x4x2_t`) | 集群（AArch64） |
| `__AVX__` | x86 AVX (`__m256`) | 本地 x86 测试 |
| 其他 | 标量数组 `float[8]` | 任意平台 |

**公共接口**（三个后端一致）：

```cpp
simd8float32()                        // 默认构造（值未定义）
explicit simd8float32(const float* x) // 从内存加载 8 个 float（非对齐安全）
static simd8float32 zeros()           // 返回全 0
simd8float32 operator+(rhs)           // 逐元素加法
simd8float32 operator*(rhs)           // 逐元素乘法
simd8float32& operator+=(rhs)         // 原地加法
simd8float32& operator*=(rhs)         // 原地乘法
void store(float* dst)                // 写回 8 个 float 到内存
float reduce_add()                    // 水平求和（所有 8 个 lane 的和）
```

**辅助函数**：

```cpp
float simd_inner_product(const float* a, const float* b, size_t dim);
// 使用 simd8float32 计算内积，自动处理非 8 倍数的尾部
```

**ARM NEON 关键 intrinsic 说明**：
- `vld1q_f32(ptr)` — 加载 4 个 float 到 128 位寄存器
- `vmulq_f32(a, b)` — 逐元素乘法
- `vaddq_f32(a, b)` — 逐元素加法
- `vst1q_f32(ptr, v)` — 存储到内存
- `vaddvq_f32(v)` — AArch64 水平求和（4 个 lane → 标量）

---

### [hnswlib/](hnswlib/)

第三方 HNSW 库，支持：
- **InnerProductSpace** / **L2Space** 距离空间
- **HierarchicalNSW** —— 分层小世界图近似搜索
  - 关键参数：`M`（每层最大邻居数），`efConstruction`（构建时搜索宽度），`ef`（查询时搜索宽度）
  - 线程安全的 `addPoint`（可配合 `#pragma omp parallel for` 并行插入）
  - `searchKnn(query, k)` → `priority_queue<{dist, label}>`（远优先）
  - `saveIndex(path)` / `loadIndex(path, space)` 持久化
- SIMD 自动选择：SSE / AVX / AVX512（运行时检测 CPU 能力）

### [qsub.sh](qsub.sh)

PBS 作业脚本，在集群计算节点上运行已编译的 `main`：
1. 从 master 节点拷贝可执行文件和 `files/` 目录到计算节点
2. 运行 `/home/${USER}/main`
3. 将结果 `files/` 目录同步回 master

**注意**：索引必须保存在 `files/` 目录下。

---

## 数据集说明

- **DEEP100K**：100K 条高维浮点向量
- **距离度量**：内积距离（Inner Product），`dis = 1 - dot(a, b)`
- **评测指标**：recall@10（top-10 召回率）vs ground truth top-100

---

## 修改约束

| 文件 | 允许操作 |
|------|---------|
| `main.cc` | 可修改搜索函数调用方式（函数名/参数/改为成员函数调用）；不可改返回值类型；不可改评测逻辑 |
| `flat_scan.h` | **禁止修改** |
| `hnswlib/` | 只读使用 |
| 新增文件 | 可自由添加头文件/源文件 |
| `files/` 目录 | 索引保存位置（每人空间有限，及时清理） |

---

## 优化思路

1. **HNSW 近似搜索**：替换 `flat_search` 调用为加载预建 HNSW 索引后的 `searchKnn`
2. **并行暴力搜索**：用 OpenMP 并行化内积计算（已包含 `<omp.h>`）
3. **SIMD 加速**：手写 AVX/SSE 内积（hnswlib 的 `space_ip.h` 有参考实现）
4. **批量查询**：一次处理多条查询，提升 cache 利用率
5. **参数调优**：HNSW 的 `ef`（查询宽度）直接影响 recall 与延迟的权衡

---

## 编译与运行

### 本地编译（仅验证编译正确性，无法运行，数据只在集群）

在 `ann/` 目录下执行：

```bash
# 基础版（标量 fallback，任意平台）
g++ main.cc -o main -O2 -fopenmp -lpthread -std=c++11

# 推荐版（启用 x86 AVX，本地有 AVX 支持时使用）
g++ main.cc -o main -O3 -march=native -fopenmp -lpthread -std=c++11
```

> Windows Git Bash 下若无 g++，可用 WSL：`wsl g++ main.cc -o main -O2 -fopenmp -lpthread -std=c++11`

### 登录集群

```bash
# 一键跳板登录集群主节点（学号 2313483）
ssh -J s2313483@10.137.144.91:9001 s2313483@192.168.90.141
```

VSCode Remote-SSH 配置（写入 `~/.ssh/config`）：

```
Host s2313483
    HostName 192.168.90.141
    ProxyJump s2313483@10.137.144.91:9001
    User s2313483
```

### 上传代码到集群

在本地 `ann/` 目录下执行，将改动的头文件和 main.cc 同步到集群：

```bash
scp -J s2313483@10.137.144.91:9001 \
    main.cc simd_utils.h simd_search.h \
    s2313483@192.168.90.141:~/ann/
```

### 集群提交测试（正式评测，必须用此方式）

SSH 登录集群后，在 `~/ann/` 目录下执行：

```bash
# 第一个参数：实验序号（1=SIMD, 2=pthread/OpenMP, ...）
# 第二个参数：申请节点数（SIMD 实验只能申请 1 个）
bash test.sh 1 1
```

脚本自动编译（使用固定命令 `g++ main.cc -o main -O2 -fopenmp -lpthread -std=c++11`）并提交 PBS 作业，结束后输出：
- `test.o` — 最终结果（`std::cout`，含 recall 和 latency）
- `test.e` — 数据加载日志（`std::cerr`）

**禁止直接运行 `./main`，否则成绩不会被记录。**

### 预期输出（SIMD 阶段）

```
average recall: 1
average latency (us): XX
```

---

## 当前实现状态（2026-04-28）

### Flat-SIMD（已完成，得分阶段）

| 文件 | 核心内容 |
|------|---------|
| `simd_utils.h` | `simd8float32`：ARM NEON float32x4x2_t 封装；`simd_inner_product`：3 路 FMA 展开 |
| `simd_search.h` | `simd_flat_search_v1–v4`：消融对比版本；`simd_flat_search` = v4（3路展开 + 软件预取） |

**最终结果**：v4 latency = 5208µs，recall = 0.99995，**3.09× vs Baseline**

### SQ-SIMD（已完成，得分阶段）

| 文件 | 核心内容 |
|------|---------|
| `sq_search.h` | `int8_int8_ip`：纯 int8×int8 SIMD 内积（NEON/AVX2/标量）；`SQIndex`：per-vector 对称量化 + 两阶段搜索 |

**关键参数**：p=19 时 recall=0.99995，latency=2587µs，**2.01× vs Flat-SIMD，6.21× vs Baseline**

**p-sweep 结论**：
- p < 10：recall ≈ 0.984（过渡区下方）
- p = 10–19：recall 快速上升至 0.99995
- p = 19–100：latency ≈ 2540–2650µs（最优区间）
- p > 1000：Phase 2 rerank 开销上升，p > 3000 时反超 Flat-SIMD

### PQ-SIMD（实现中）

| 文件 | 核心内容 |
|------|---------|
| `generate_pq.py` | MiniBatchKMeans 生成码本；输出 `files/pq_codebook.bin`（4×256×24 f32）和 `files/pq_codes.bin`（N×4 u8）|
| `pq_search.h` | `pq_l2sq_24`：3路 NEON 累加器；`PQIndex`：load+LUT 构建+两阶段搜索 |

**使用流程**：
1. 集群上先运行 `python3 generate_pq.py`（约 1–2 分钟）生成码本
2. 编译并提交：`bash test.sh 1 1`

**p-sweep 范围**：p=1..50 逐个 + 75,100,150,200,300,500,750,1000,1500,2000,3000,5000

**输出格式**：`p=X latency=Yms recall=Z`

### main.cc 当前状态

当前为 **PQ p-sweep 模式**。若需切回 SQ，在 load() 后改为调用 `sq_idx.search()`。

---

## 更新记录

| 日期 | 内容 |
|------|------|
| 2026-04-27 | 初始 CLAUDE.md，记录项目结构与核心代码分析 |
| 2026-04-27 | 新增 `simd_utils.h`：`simd8float32` 封装（3路后端）及 `simd_inner_product` |
| 2026-04-27 | 完善 `simd_utils.h`：FMA、广播构造、4路展开；新增 `simd_search.h` |
| 2026-04-28 | `simd_search.h` 重构为 v1–v4 消融版本；分析 3 路展开优于 4 路的原因 |
| 2026-04-28 | 新增 `sq_search.h`：SQIndex，int8×int8 SIMD 两阶段搜索；p-sweep 分析完成 |
| 2026-04-28 | 新增 `generate_pq.py` + `pq_search.h`：PQ-SIMD 两阶段搜索；main.cc 切换到 PQ p-sweep |
