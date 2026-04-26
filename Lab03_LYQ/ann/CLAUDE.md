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
├── my_test          # 已编译的 ARM64 ELF 二进制（测试用可执行文件）
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

```bash
# 编译（参考，具体 Makefile 见集群环境）
g++ -O3 -march=native -fopenmp -o main main.cc -I.

# 本地测试（需要 /anndata/ 数据）
./main

# 集群提交
qsub qsub.sh
```

---

## 更新记录

| 日期 | 内容 |
|------|------|
| 2026-04-27 | 初始 CLAUDE.md，记录项目结构与核心代码分析 |
