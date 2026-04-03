"""
visualize_exp1.py  —  实验一结果可视化 + 表格

读取两个 CSV：
  results/exp1_cache.csv  -> 图1：时间+加速比（双Y轴，-O2）
  results/exp1_opt.csv    -> 表格：编译器优化等级效果（输出到终端 + txt文件）

运行：python scripts/visualize_exp1.py
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT         = os.path.dirname(SCRIPT_DIR)
CSV_CACHE    = os.path.join(ROOT, "results", "exp1_cache.csv")
CSV_OPT      = os.path.join(ROOT, "results", "exp1_opt.csv")
FIG_DIR      = os.path.join(ROOT, "results", "figures")
TABLE_OUT    = os.path.join(ROOT, "results", "exp1_opt_table.txt")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize":10,
    "figure.dpi":     150,
    "axes.grid":      True,
    "grid.alpha":     0.3,
    "grid.linestyle": "--",
})

# ─────────────────────────────────────────────────
# 图1：双Y轴，cache优化实验（固定-O2）
# ─────────────────────────────────────────────────
def plot_cache(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["correct"] == "OK"].copy()
    df["avg_naive_ms"] = df["t_naive_ms"] / df["runs"]
    df["avg_opt_ms"]   = df["t_opt_ms"]   / df["runs"]
    df["speedup"]      = df["avg_naive_ms"] / df["avg_opt_ms"]
    df = df.sort_values("n")

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    l1, = ax1.plot(df["n"], df["avg_naive_ms"],
                   color="#E74C3C", marker="o", ms=3, lw=1.8, label="Naive (time)")
    l2, = ax1.plot(df["n"], df["avg_opt_ms"],
                   color="#2980B9", marker="s", ms=3, lw=1.8, label="Cache-Opt (time)")
    l3, = ax2.plot(df["n"], df["speedup"],
                   color="#27AE60", marker="^", ms=3, lw=1.5,
                   linestyle="--", label="Speedup (×)")
    ax2.axhline(1.0, color="gray", lw=1, linestyle=":", alpha=0.6)

    ax1.set_xlabel("Matrix size n")
    ax1.set_ylabel("Avg time per run (ms)")
    ax2.set_ylabel("Speedup (Naive / Cache-Opt)", color="#27AE60")
    ax2.tick_params(axis="y", labelcolor="#27AE60")
    ax1.legend([l1, l2, l3], [l.get_label() for l in [l1, l2, l3]], loc="upper left")
    ax1.set_title("Exp1: Execution Time & Speedup vs Matrix Size  (compiler: -O2)")
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig1_time_speedup.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"  saved: {out}")
    plt.close(fig)

# ─────────────────────────────────────────────────
# 表格：编译器优化等级实验
# 格式：n | runs | Naive_O0 | Naive_O1 | Naive_O2 | Naive_O3 | Opt_O0 | ... | Speedup_O2
# ─────────────────────────────────────────────────
def print_opt_table(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["correct"] == "OK"].copy()
    df["avg_naive_ms"] = (df["t_naive_ms"] / df["runs"]).round(6)
    df["avg_opt_ms"]   = (df["t_opt_ms"]   / df["runs"]).round(6)

    OPT_LEVELS = ["O0", "O1", "O2", "O3"]

    # pivot：行=n，列=opt_label
    naive_pivot = df.pivot_table(index="n", columns="opt_label", values="avg_naive_ms")
    opt_pivot   = df.pivot_table(index="n", columns="opt_label", values="avg_opt_ms")
    runs_s      = df[df["opt_label"] == "O2"].set_index("n")["runs"]

    # 对齐列顺序
    naive_pivot = naive_pivot.reindex(columns=OPT_LEVELS)
    opt_pivot   = opt_pivot.reindex(columns=OPT_LEVELS)

    ns = sorted(naive_pivot.index)

    # ── 表1：Naive算法，各优化等级平均时间 ──────────
    lines = []
    header = f"{'n':>7} | {'runs':>6} | " + " | ".join(f"Naive-{o}(ms)" for o in OPT_LEVELS)
    sep    = "-" * len(header)
    lines.append("=== Table 1: Naive Algorithm — Avg time per run (ms) ===")
    lines.append(sep)
    lines.append(header)
    lines.append(sep)
    for n in ns:
        row = f"{n:>7} | {int(runs_s.get(n, 0)):>6} | "
        row += " | ".join(
            f"{naive_pivot.loc[n, o]:>13.6f}" if o in naive_pivot.columns and not pd.isna(naive_pivot.loc[n, o])
            else f"{'N/A':>13}"
            for o in OPT_LEVELS
        )
        lines.append(row)
    lines.append(sep)

    lines.append("")

    # ── 表2：Cache-Opt算法，各优化等级平均时间 ───────
    header2 = f"{'n':>7} | {'runs':>6} | " + " | ".join(f"Opt-{o}(ms)" for o in OPT_LEVELS)
    sep2    = "-" * len(header2)
    lines.append("=== Table 2: Cache-Opt Algorithm — Avg time per run (ms) ===")
    lines.append(sep2)
    lines.append(header2)
    lines.append(sep2)
    for n in ns:
        row = f"{n:>7} | {int(runs_s.get(n, 0)):>6} | "
        row += " | ".join(
            f"{opt_pivot.loc[n, o]:>11.6f}" if o in opt_pivot.columns and not pd.isna(opt_pivot.loc[n, o])
            else f"{'N/A':>11}"
            for o in OPT_LEVELS
        )
        lines.append(row)
    lines.append(sep2)

    output = "\n".join(lines)
    print(output)

    with open(TABLE_OUT, "w", encoding="utf-8") as f:
        f.write(output + "\n")
    print(f"\n  表格已保存至: {TABLE_OUT}")

# ── 主入口 ────────────────────────────────────────
if __name__ == "__main__":
    for path, label in [(CSV_CACHE, "exp1_cache.csv"), (CSV_OPT, "exp1_opt.csv")]:
        if not os.path.exists(path):
            print(f"[ERROR] 找不到: {path}，请先运行 run_exp1.sh")
            sys.exit(1)

    print("生成图1（cache优化实验）...")
    plot_cache(CSV_CACHE)

    print("\n编译器优化等级结果表格：")
    print_opt_table(CSV_OPT)

    print(f"\n完成。图表: {FIG_DIR}/fig1_time_speedup.png")