#!/usr/bin/env bash
# ============================================================
# run_exp1.sh  —  实验一：编译 + 批量测试 + 输出 CSV
#
# 输出两个 CSV：
#   results/exp1_cache.csv  — cache优化实验（分段线性规模，固定-O2）
#   results/exp1_opt.csv    — 编译器优化等级实验（固定规模/次数组）
#
# 用法（Git Bash）：bash run_exp1.sh
# ============================================================

set -uo pipefail

export PATH="/c/Program Files (x86)/Dev-Cpp/MinGW64/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/src/exp1.cpp"
BUILD="$SCRIPT_DIR/build"
RESULTS="$SCRIPT_DIR/results"
CSV_CACHE="$RESULTS/exp1_cache.csv"
CSV_OPT="$RESULTS/exp1_opt.csv"

mkdir -p "$BUILD" "$RESULTS"

# ── 1. 编译四个优化等级 ───────────────────────────
echo "[1/4] 编译..."
for OPT in O0 O1 O2 O3; do
    echo "  g++ -${OPT} -> build/exp1_${OPT}.exe"
    g++ -std=c++11 -"${OPT}" -o "$BUILD/exp1_${OPT}.exe" "$SRC" || { echo "[ERROR] 编译 -${OPT} 失败"; exit 1; }
done
echo "  编译完成"

# ── 2. cache优化实验：固定-O2，分段线性规模 ──────────
echo "opt_label,n,runs,t_naive_ms,t_opt_ms,correct" > "$CSV_CACHE"

get_runs() {
    local n=$1
    if   (( n <= 64   )); then echo 500
    elif (( n <= 256  )); then echo 100
    elif (( n <= 512  )); then echo 30
    elif (( n <= 1024 )); then echo 10
    else                       echo 3
    fi
}

N_LIST=()
for (( n=16;  n<=128;  n+=16 )); do N_LIST+=($n); done
for (( n=144; n<=512;  n+=16 )); do N_LIST+=($n); done
for (( n=576; n<=2048; n+=64 )); do N_LIST+=($n); done

echo "[2/4] cache优化实验（-O2，${#N_LIST[@]} 规模点）..."
DONE=0
for n in "${N_LIST[@]}"; do
    runs=$(get_runs "$n")
    result=$("$BUILD/exp1_O2.exe" "$n" "$runs" "O2" || true)
    echo "$result" >> "$CSV_CACHE"
    (( DONE++ ))
    if (( DONE % 10 == 0 || DONE == ${#N_LIST[@]} )); then
        echo "  进度: ${DONE}/${#N_LIST[@]}  (n=${n})"
    fi
done
echo "  完成，结果保存至: $CSV_CACHE"

# ── 3. 编译器优化等级实验：固定规模/次数组，跑4个等级 ─
echo "opt_label,n,runs,t_naive_ms,t_opt_ms,correct" > "$CSV_OPT"

OPT_N=(   0    10    20    50   100   150   200   500  1000  2000  3000  5000  10000  20000)
OPT_RUNS=(10000 10000 10000 5000 1000  1000  500   100   10    10    1     1     1      1)

TOTAL_OPT=$(( 4 * ${#OPT_N[@]} ))
DONE=0
echo "[3/4] 编译器优化等级实验（4等级 × ${#OPT_N[@]} 规模 = $TOTAL_OPT 组）..."
for OPT in O0 O1 O2 O3; do
    for i in "${!OPT_N[@]}"; do
        n="${OPT_N[$i]}"
        runs="${OPT_RUNS[$i]}"
        result=$("$BUILD/exp1_${OPT}.exe" "$n" "$runs" "$OPT" || true)
        echo "$result" >> "$CSV_OPT"
        (( DONE++ ))
        if (( DONE % 10 == 0 || DONE == TOTAL_OPT )); then
            echo "  进度: ${DONE}/${TOTAL_OPT}  (opt=${OPT}, n=${n})"
        fi
    done
done
echo "  完成，结果保存至: $CSV_OPT"

echo "[4/4] 全部完成！"
echo "  cache实验: $CSV_CACHE"
echo "  优化等级:  $CSV_OPT"