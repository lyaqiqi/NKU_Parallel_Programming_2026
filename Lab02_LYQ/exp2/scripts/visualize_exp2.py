"""
Lab02 实验二可视化脚本

输入：
  exp2_perf.csv      - 各算法 × 规模的执行时间
  exp2_opt.csv       - 编译器优化等级对比（可选）
  exp2_perf_log.txt  - perf stat 原始输出（可选）

输出：
  fig_exp2_time.png  - 图：执行时间 + 加速比（双子图）
  fig_exp2_perf.png  - 图：perf IPC + branch-miss 柱状图（有 perf_log 才生成）
  终端打印：编译器优化等级表格、perf 汇总表格
"""

import os, re, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 路径 ────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def find(name):
    p = os.path.join(ROOT, name)
    if not os.path.exists(p):
        print(f"[WARN] 找不到 {p}，跳过相关输出")
        return None
    return p

PERF_CSV = find("exp2_perf.csv")
OPT_CSV  = find("exp2_opt.csv")
PERF_LOG = find("exp2_perf_log.txt")
OUT_DIR  = ROOT

# ── 样式 ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize":9,
    "axes.grid":      True,
    "grid.alpha":     0.3,
    "grid.linestyle": "--",
    "figure.dpi":     150,
})

# 算法显示顺序、颜色、线型
ALGO_ORDER = ["naive", "unroll2", "unroll4", "unroll8", "unroll16", "unroll32", "recursive"]
ALGO_META  = {
    "naive":     {"label": "Naive",               "color": "#4c72b0", "ls": "-",  "marker": "o"},
    "unroll2":   {"label": "Unroll 2-way",        "color": "#dd8452", "ls": "--", "marker": "s"},
    "unroll4":   {"label": "Unroll 4-way",        "color": "#55a868", "ls": "--", "marker": "^"},
    "unroll8":   {"label": "Unroll 8-way",        "color": "#c44e52", "ls": "--", "marker": "D"},
    "unroll16":  {"label": "Unroll 16-way",       "color": "#8172b2", "ls": "--", "marker": "P"},
    "unroll32":  {"label": "Unroll 32-way",       "color": "#937860", "ls": "--", "marker": "X"},
    "recursive": {"label": "Recursive",           "color": "#da8bc3", "ls": ":",  "marker": "v"},
}

def fmt_n(n):
    if n >= 1e8: return f"{int(n//1e6)}M"
    if n >= 1e6: return f"{int(n//1e6)}M"
    if n >= 1e5: return f"{int(n//1e3)}K"
    if n >= 1e3: return f"{int(n//1e3)}K"
    return str(int(n))

# ── 图1：执行时间 + 加速比 ───────────────────────────────────────
def plot_time():
    if PERF_CSV is None:
        return
    df = pd.read_csv(PERF_CSV)
    df["us_per_call"] = df["time_ms"] / df["runs"] * 1000   # μs

    ns    = sorted(df["n"].unique())
    algos = [a for a in ALGO_ORDER if a in df["algo"].unique()]
    xs    = range(len(ns))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Exp2: Sum of N floats — Timing Comparison (-O2)", y=1.02)

    # 左：绝对时间（log 轴）
    for algo in algos:
        sub = df[df["algo"] == algo].sort_values("n")
        m   = ALGO_META[algo]
        ax1.plot(xs, sub["us_per_call"].values,
                 label=m["label"], color=m["color"],
                 ls=m["ls"], marker=m["marker"], markersize=6, lw=1.7)
    ax1.set_xticks(xs); ax1.set_xticklabels([fmt_n(n) for n in ns])
    ax1.set_xlabel("Array size n")
    ax1.set_ylabel("Time per call (μs)")
    ax1.set_title("Execution Time (log scale)")
    ax1.set_yscale("log")
    ax1.legend(loc="upper left", framealpha=0.9)

    # 右：加速比（相对 naive）
    naive_t = df[df["algo"] == "naive"].sort_values("n")["us_per_call"].values
    for algo in algos:
        if algo == "naive":
            continue
        sub     = df[df["algo"] == algo].sort_values("n")
        speedup = naive_t / sub["us_per_call"].values
        m       = ALGO_META[algo]
        ax2.plot(xs, speedup,
                 label=m["label"], color=m["color"],
                 ls=m["ls"], marker=m["marker"], markersize=6, lw=1.7)
    ax2.axhline(1.0, color="gray", ls=":", lw=1.2, label="Baseline (Naive)")
    ax2.set_xticks(xs); ax2.set_xticklabels([fmt_n(n) for n in ns])
    ax2.set_xlabel("Array size n")
    ax2.set_ylabel("Speedup over Naive")
    ax2.set_title("Speedup Ratio")
    ax2.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig_exp2_time.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[OK] {out}")
    plt.close(fig)

# ── 解析 perf_log ────────────────────────────────────────────────
def _parse_num(s):
    """把 '1,234,567' 或 '1.234.567' 之类的数字字符串转 float"""
    s = s.strip()
    # 千分位可能是逗号或点，去掉后转换
    s = re.sub(r"[,.](?=\d{3})", "", s)
    return float(s.replace(",", "."))

def parse_perf_log():
    if PERF_LOG is None:
        return None
    text = open(PERF_LOG).read()
    blocks = re.split(r"={8} algo=(\w+)", text)
    records = []
    for i in range(1, len(blocks), 2):
        algo = blocks[i].strip()
        body = blocks[i+1]

        def grab(pat):
            # 跳过 <not supported> 行
            for line in body.splitlines():
                if re.search(pat, line) and "<not supported>" not in line:
                    m = re.search(r"([\d,\.]+)", line)
                    if m:
                        try: return _parse_num(m.group(1))
                        except: return None
            return None

        cycles  = grab(r"cycles")
        instruc = grab(r"instructions")
        dcm     = grab(r"L1-dcache-load-misses")
        icm     = grab(r"L1-icache-load-misses")
        bm      = grab(r"branch-misses")
        # 软件事件
        cpu_clk = grab(r"cpu-clock|task-clock")

        ipc_m = re.search(r"([\d.]+)\s+insn per cycle", body)
        if ipc_m:
            ipc = float(ipc_m.group(1))
        elif cycles and instruc and cycles > 0:
            ipc = instruc / cycles
        else:
            ipc = None

        records.append(dict(algo=algo, cycles=cycles, instructions=instruc,
                            IPC=ipc, dcache_miss=dcm, icache_miss=icm,
                            branch_miss=bm, cpu_clock=cpu_clk))

    if not records:
        return None
    df = pd.DataFrame(records)
    # 如果所有硬件指标都是 None，说明是软件事件模式
    hw_cols = ["cycles", "instructions", "IPC"]
    df.attrs["hw_available"] = not df[hw_cols].isnull().all().all()
    return df

# ── 图2：perf 柱状图（硬件事件 or 软件事件自适应）──────────────
def plot_perf():
    df = parse_perf_log()
    if df is None or df.empty:
        print("[WARN] perf 数据不可用，跳过 fig_exp2_perf.png")
        return

    hw = df.attrs.get("hw_available", False)
    algos  = [a for a in ALGO_ORDER if a in df["algo"].values]
    labels = [ALGO_META[a]["label"] for a in algos]
    colors = [ALGO_META[a]["color"] for a in algos]
    x      = np.arange(len(algos))

    if hw:
        # 硬件事件：IPC + icache-miss/cycle
        ipc = [df[df["algo"]==a]["IPC"].values[0] or 0 for a in algos]
        icm_rate = []
        for a in algos:
            row = df[df["algo"]==a].iloc[0]
            icm = row["icache_miss"] or 0
            cyc = row["cycles"]      or 1
            icm_rate.append(icm / cyc * 100)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        fig.suptitle("Exp2: perf Profiling Results (Hardware Events)", y=1.02)

        bars = ax1.bar(x, ipc, color=colors, width=0.6, edgecolor="white")
        ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=20, ha="right")
        ax1.set_ylabel("IPC (Instructions Per Cycle)")
        ax1.set_title("IPC — higher is better")
        for bar, v in zip(bars, ipc):
            if v: ax1.text(bar.get_x()+bar.get_width()/2, v+0.005,
                           f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)

        bars2 = ax2.bar(x, icm_rate, color=colors, width=0.6, edgecolor="white")
        ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=20, ha="right")
        ax2.set_ylabel("L1-icache-miss / cycle (%)")
        ax2.set_title("Instruction Cache Pressure")
        for bar, v in zip(bars2, icm_rate):
            ax2.text(bar.get_x()+bar.get_width()/2, v+0.0001,
                     f"{v:.4f}%", ha="center", va="bottom", fontsize=8.5)
    else:
        # 软件事件降级：只画 cpu-clock（执行时间代理指标）
        clk = [df[df["algo"]==a]["cpu_clock"].values[0] or 0 for a in algos]
        # 归一化到 naive
        base = clk[0] if clk[0] > 0 else 1.0
        rel  = [v / base for v in clk]

        fig, ax1 = plt.subplots(1, 1, figsize=(7, 4.5))
        fig.suptitle("Exp2: perf cpu-clock (Software Event, Normalized to Naive)", y=1.02)
        bars = ax1.bar(x, rel, color=colors, width=0.6, edgecolor="white")
        ax1.axhline(1.0, color="gray", ls=":", lw=1.2)
        ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=20, ha="right")
        ax1.set_ylabel("Relative cpu-clock (lower is better)")
        ax1.set_title("注：硬件 PMU 在此环境不可用，仅供参考")
        for bar, v in zip(bars, rel):
            ax1.text(bar.get_x()+bar.get_width()/2, v+0.01,
                     f"{v:.2f}x", ha="center", va="bottom", fontsize=8.5)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig_exp2_perf.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[OK] {out}")
    plt.close(fig)

# ── 表格：perf 汇总 ──────────────────────────────────────────────
def table_perf():
    df = parse_perf_log()
    if df is None or df.empty:
        return
    hw    = df.attrs.get("hw_available", False)
    algos = [a for a in ALGO_ORDER if a in df["algo"].values]

    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        return f"{int(v):,}"

    if hw:
        print("\n" + "="*88)
        print("perf 事件计数汇总（硬件事件，固定规模）")
        print("="*88)
        print(f"  {'算法':<24} {'IPC':>7}  {'cycles':>13}  {'instructions':>13}  {'dcache-miss':>12}  {'icache-miss':>12}  {'branch-miss':>12}")
        print("  " + "-"*93)
        for a in algos:
            row   = df[df["algo"]==a].iloc[0]
            ipc_s = f"{row['IPC']:.4f}" if row["IPC"] else "N/A"
            print(f"  {ALGO_META[a]['label']:<24} {ipc_s:>7}  {fmt(row['cycles']):>13}  "
                  f"{fmt(row['instructions']):>13}  {fmt(row['dcache_miss']):>12}  "
                  f"{fmt(row['icache_miss']):>12}  {fmt(row['branch_miss']):>12}")
        print("="*88 + "\n")
    else:
        print("\n" + "="*60)
        print("perf 软件事件汇总（硬件 PMU 在此环境不可用）")
        print("="*60)
        print(f"  {'算法':<24} {'cpu-clock (ms)':>16}  {'相对 naive':>12}")
        print("  " + "-"*55)
        base = df[df["algo"]=="naive"]["cpu_clock"].values
        base = base[0] if len(base) and base[0] else 1.0
        for a in algos:
            row = df[df["algo"]==a].iloc[0]
            clk = row["cpu_clock"]
            clk_s = f"{clk:.2f}" if clk else "N/A"
            rel_s = f"{clk/base:.3f}x" if clk else "N/A"
            print(f"  {ALGO_META[a]['label']:<24} {clk_s:>16}  {rel_s:>12}")
        print("="*60 + "\n")

# ── 表格：编译器优化等级 ─────────────────────────────────────────
def table_opt():
    if OPT_CSV is None:
        return
    df = pd.read_csv(OPT_CSV)
    df["us_per_call"] = df["time_ms"] / df["runs"] * 1000

    print("\n" + "="*62)
    print("编译器优化等级对比（algo=naive）")
    print("="*62)
    print(f"  {'n':>12}  {'opt':>4}  {'μs/call':>12}  {'speedup vs O0':>14}")
    print(f"  {'-'*12}  {'-'*4}  {'-'*12}  {'-'*14}")
    for n_val, grp in df.groupby("n", sort=True):
        base_rows = grp[grp["opt_level"]=="O0"]["us_per_call"].values
        base = base_rows[0] if len(base_rows) else 1.0
        for _, row in grp.sort_values("opt_level").iterrows():
            t  = row["us_per_call"]
            sp = base / t if t > 0 else float("nan")
            print(f"  {int(n_val):>12,}  {row['opt_level']:>4}  {t:>12.2f}  {sp:>14.2f}x")
    print("="*62 + "\n")

# ── 主入口 ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[INFO] 生成实验二图表和表格...")
    plot_time()
    plot_perf()
    table_perf()
    table_opt()
    print("[INFO] 完成。")