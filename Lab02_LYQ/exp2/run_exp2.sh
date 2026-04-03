#!/bin/bash
# Lab02 实验二测试脚本
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$SCRIPT_DIR/src/exp2.cpp"
BUILD="$SCRIPT_DIR/build"
mkdir -p "$BUILD"

GREEN='\033[0;32m'; NC='\033[0m'
info() { echo -e "${GREEN}[INFO]${NC}  $*"; }

CXX=${CXX:-g++}
info "编译器: $($CXX --version | head -1)"

info "编译中..."
$CXX -std=c++11 -O2 -march=native -o "$BUILD/exp2" "$SRC"
info "  $BUILD/exp2"

ALGOS="naive unroll2 unroll4 unroll8 unroll16 unroll32 recursive"

# ── 实验A：整数性能对比 ───────────────────────────────────────
PERF_CSV="$SCRIPT_DIR/exp2_perf.csv"
echo "algo,n,runs,time_ms,result" > "$PERF_CSV"
info "=== 实验A：整数算法性能对比（-O2 -march=native）==="

# 10 个规模，runs 随规模递减
declare -A RUNS
RUNS[8192]=5000
RUNS[16384]=2000
RUNS[32768]=1000
RUNS[65536]=500
RUNS[131072]=200
RUNS[262144]=100
RUNS[524288]=50
RUNS[1048576]=20
RUNS[4194304]=5
RUNS[67108864]=1

for N in 8192 16384 32768 65536 131072 262144 524288 1048576 4194304 67108864; do
    R=${RUNS[$N]}
    for ALGO in $ALGOS; do
        info "  $ALGO  n=$N  runs=$R"
        "$BUILD/exp2" perf "$N" "$R" "$ALGO" >> "$PERF_CSV"
    done
done
info "=> $PERF_CSV"

# ── 实验B：浮点精度探索 ───────────────────────────────────────
FLOAT_CSV="$SCRIPT_DIR/exp2_float.csv"
echo "n,naive,unroll2,unroll4,unroll8,unroll16,unroll32,ref_double,max_err" > "$FLOAT_CSV"
info "=== 实验B：浮点精度探索 ==="

for N in 32 256 1024 8192 65536 524288 4194304; do
    info "  float  n=$N"
    "$BUILD/exp2" float "$N" >> "$FLOAT_CSV"
done
info "=> $FLOAT_CSV"

info "=== 完成 ==="