/**
 * Lab02 实验二：n 个数求和（超标量 / 循环展开优化）
 *
 * 用法：
 *   ./exp2 perf  <n> <runs> <algo>    -- 整数性能测试
 *   ./exp2 float <n>                  -- 浮点精度探索
 *
 * 整数性能输出（CSV）：algo,n,runs,time_ms,result
 * 浮点精度输出（CSV）：n,naive,unroll2,unroll4,unroll8,unroll16,unroll32,max_diff
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <time.h>

static double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ── 整数数组初始化 ── */
static void fill_int(int* a, int n) {
    for (int i = 0; i < n; i++) a[i] = i % 1024 + 1;
}

/* ── 浮点数组初始化：使用差异较大的值以放大精度差异 ── */
static void fill_float(float* a, int n) {
    // 混合大数和小数，容易暴露累加顺序导致的精度差
    for (int i = 0; i < n; i++) {
        if (i % 16 == 0)
            a[i] = 1e6f;        // 大数
        else
            a[i] = 0.1f;        // 小数（二进制下不能精确表示）
    }
}

/* ════════════════════════════════════════════
   整数求和算法
   ════════════════════════════════════════════ */

int common_algo(int *arr, int len) {
    int total = 0;
    for (int i = 0; i < len; i++) total += arr[i];
    return total;
}

int sum_naive(int *a, int n) {
    int s = 0;
    for (int i = 0; i < n; i++) s += a[i];
    return s;
}

int sum_unroll2(int *a, int n) {
    int t[2] = {0};
    for (int i = 0; i < n; i += 2) { t[0] += a[i]; t[1] += a[i+1]; }
    return common_algo(t, 2);
}

int sum_unroll4(int *a, int n) {
    int t[4] = {0};
    for (int i = 0; i < n; i += 4) {
        t[0] += a[i]; t[1] += a[i+1]; t[2] += a[i+2]; t[3] += a[i+3];
    }
    return common_algo(t, 4);
}

int sum_unroll8(int *a, int n) {
    int t[8] = {0};
    for (int i = 0; i < n; i += 8) {
        t[0]+=a[i];   t[1]+=a[i+1]; t[2]+=a[i+2]; t[3]+=a[i+3];
        t[4]+=a[i+4]; t[5]+=a[i+5]; t[6]+=a[i+6]; t[7]+=a[i+7];
    }
    return common_algo(t, 8);
}

int sum_unroll16(int *a, int n) {
    int t[16] = {0};
    for (int i = 0; i < n; i += 16) {
        t[0] +=a[i];    t[1] +=a[i+1];  t[2] +=a[i+2];  t[3] +=a[i+3];
        t[4] +=a[i+4];  t[5] +=a[i+5];  t[6] +=a[i+6];  t[7] +=a[i+7];
        t[8] +=a[i+8];  t[9] +=a[i+9];  t[10]+=a[i+10]; t[11]+=a[i+11];
        t[12]+=a[i+12]; t[13]+=a[i+13]; t[14]+=a[i+14]; t[15]+=a[i+15];
    }
    return common_algo(t, 16);
}

int sum_unroll32(int *a, int n) {
    int t[32] = {0};
    for (int i = 0; i < n; i += 32) {
        t[0] +=a[i];    t[1] +=a[i+1];  t[2] +=a[i+2];  t[3] +=a[i+3];
        t[4] +=a[i+4];  t[5] +=a[i+5];  t[6] +=a[i+6];  t[7] +=a[i+7];
        t[8] +=a[i+8];  t[9] +=a[i+9];  t[10]+=a[i+10]; t[11]+=a[i+11];
        t[12]+=a[i+12]; t[13]+=a[i+13]; t[14]+=a[i+14]; t[15]+=a[i+15];
        t[16]+=a[i+16]; t[17]+=a[i+17]; t[18]+=a[i+18]; t[19]+=a[i+19];
        t[20]+=a[i+20]; t[21]+=a[i+21]; t[22]+=a[i+22]; t[23]+=a[i+23];
        t[24]+=a[i+24]; t[25]+=a[i+25]; t[26]+=a[i+26]; t[27]+=a[i+27];
        t[28]+=a[i+28]; t[29]+=a[i+29]; t[30]+=a[i+30]; t[31]+=a[i+31];
    }
    return common_algo(t, 32);
}

int sum_recursive(int *a, int n) {
    for (int m = n; m > 1; m /= 2)
        for (int i = 0; i < m/2; i++)
            a[i] = a[2*i] + a[2*i+1];
    return a[0];
}

/* ════════════════════════════════════════════
   浮点求和算法（float 版，结构与整数版一致）
   ════════════════════════════════════════════ */

float common_algo_f(float *arr, int len) {
    float total = 0.0f;
    for (int i = 0; i < len; i++) total += arr[i];
    return total;
}

float fsum_naive(float *a, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i];
    return s;
}

float fsum_unroll2(float *a, int n) {
    float t[2] = {0};
    for (int i = 0; i < n; i += 2) { t[0]+=a[i]; t[1]+=a[i+1]; }
    return common_algo_f(t, 2);
}

float fsum_unroll4(float *a, int n) {
    float t[4] = {0};
    for (int i = 0; i < n; i += 4) {
        t[0]+=a[i]; t[1]+=a[i+1]; t[2]+=a[i+2]; t[3]+=a[i+3];
    }
    return common_algo_f(t, 4);
}

float fsum_unroll8(float *a, int n) {
    float t[8] = {0};
    for (int i = 0; i < n; i += 8) {
        t[0]+=a[i];   t[1]+=a[i+1]; t[2]+=a[i+2]; t[3]+=a[i+3];
        t[4]+=a[i+4]; t[5]+=a[i+5]; t[6]+=a[i+6]; t[7]+=a[i+7];
    }
    return common_algo_f(t, 8);
}

float fsum_unroll16(float *a, int n) {
    float t[16] = {0};
    for (int i = 0; i < n; i += 16) {
        t[0] +=a[i];    t[1] +=a[i+1];  t[2] +=a[i+2];  t[3] +=a[i+3];
        t[4] +=a[i+4];  t[5] +=a[i+5];  t[6] +=a[i+6];  t[7] +=a[i+7];
        t[8] +=a[i+8];  t[9] +=a[i+9];  t[10]+=a[i+10]; t[11]+=a[i+11];
        t[12]+=a[i+12]; t[13]+=a[i+13]; t[14]+=a[i+14]; t[15]+=a[i+15];
    }
    return common_algo_f(t, 16);
}

float fsum_unroll32(float *a, int n) {
    float t[32] = {0};
    for (int i = 0; i < n; i += 32) {
        t[0] +=a[i];    t[1] +=a[i+1];  t[2] +=a[i+2];  t[3] +=a[i+3];
        t[4] +=a[i+4];  t[5] +=a[i+5];  t[6] +=a[i+6];  t[7] +=a[i+7];
        t[8] +=a[i+8];  t[9] +=a[i+9];  t[10]+=a[i+10]; t[11]+=a[i+11];
        t[12]+=a[i+12]; t[13]+=a[i+13]; t[14]+=a[i+14]; t[15]+=a[i+15];
        t[16]+=a[i+16]; t[17]+=a[i+17]; t[18]+=a[i+18]; t[19]+=a[i+19];
        t[20]+=a[i+20]; t[21]+=a[i+21]; t[22]+=a[i+22]; t[23]+=a[i+23];
        t[24]+=a[i+24]; t[25]+=a[i+25]; t[26]+=a[i+26]; t[27]+=a[i+27];
        t[28]+=a[i+28]; t[29]+=a[i+29]; t[30]+=a[i+30]; t[31]+=a[i+31];
    }
    return common_algo_f(t, 32);
}

/* ════════════════════════════════════════════
   main
   ════════════════════════════════════════════ */

typedef int   (*ifn_t)(int*,   int);
typedef float (*ffn_t)(float*, int);

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage:\n");
        fprintf(stderr, "  %s perf  <n> <runs> <algo>\n", argv[0]);
        fprintf(stderr, "  %s float <n>\n", argv[0]);
        return 1;
    }

    /* ── 模式1：整数性能测试 ── */
    if (!strcmp(argv[1], "perf")) {
        if (argc < 5) { fprintf(stderr, "perf needs n runs algo\n"); return 1; }
        int n    = atoi(argv[2]);
        int runs = atoi(argv[3]);
        const char* algo = argv[4];

        int* a    = (int*)malloc(n * sizeof(int));
        int* work = (int*)malloc(n * sizeof(int));
        fill_int(a, n);

        ifn_t f = NULL;
        if      (!strcmp(algo, "naive"))     f = sum_naive;
        else if (!strcmp(algo, "unroll2"))   f = sum_unroll2;
        else if (!strcmp(algo, "unroll4"))   f = sum_unroll4;
        else if (!strcmp(algo, "unroll8"))   f = sum_unroll8;
        else if (!strcmp(algo, "unroll16"))  f = sum_unroll16;
        else if (!strcmp(algo, "unroll32"))  f = sum_unroll32;
        else if (!strcmp(algo, "recursive")) f = sum_recursive;
        else { fprintf(stderr, "Unknown algo: %s\n", algo); return 1; }

        int result = 0;
        double t0 = now_ms();
        if (!strcmp(algo, "recursive")) {
            for (int r = 0; r < runs; r++) {
                memcpy(work, a, n * sizeof(int));
                result = f(work, n);
            }
        } else {
            for (int r = 0; r < runs; r++) result = f(a, n);
        }
        double t1 = now_ms();
        volatile int sink = result; (void)sink;

        printf("%s,%d,%d,%.6f,%d\n", algo, n, runs, t1 - t0, result);
        free(a); free(work);

    /* ── 模式2：浮点精度探索 ── */
    } else if (!strcmp(argv[1], "float")) {
        if (argc < 3) { fprintf(stderr, "float needs n\n"); return 1; }
        int n = atoi(argv[2]);
        // n 必须是 32 的倍数
        n = (n / 32) * 32;
        if (n == 0) { fprintf(stderr, "n too small\n"); return 1; }

        float* a = (float*)malloc(n * sizeof(float));
        fill_float(a, n);

        // 各算法结果
        float r_naive    = fsum_naive(a, n);
        float r_unroll2  = fsum_unroll2(a, n);
        float r_unroll4  = fsum_unroll4(a, n);
        float r_unroll8  = fsum_unroll8(a, n);
        float r_unroll16 = fsum_unroll16(a, n);
        float r_unroll32 = fsum_unroll32(a, n);

        // 用 double 计算参考值
        double ref = 0.0;
        for (int i = 0; i < n; i++) ref += (double)a[i];

        // 各算法与参考值的绝对误差
        float vals[6] = {r_naive, r_unroll2, r_unroll4, r_unroll8, r_unroll16, r_unroll32};
        float max_err = 0.0f;
        for (int i = 0; i < 6; i++) {
            float e = fabsf(vals[i] - (float)ref);
            if (e > max_err) max_err = e;
        }

        // 输出：n, 各算法结果, 参考值, 最大误差
        printf("%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.6f,%.2f\n",
               n,
               r_naive, r_unroll2, r_unroll4, r_unroll8, r_unroll16, r_unroll32,
               (float)ref, max_err);

        free(a);
    } else {
        fprintf(stderr, "Unknown mode: %s\n", argv[1]);
        return 1;
    }

    return 0;
}