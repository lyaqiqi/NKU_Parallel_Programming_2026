/**
 * exp1.cpp  —  实验一：矩阵列向量内积
 *
 * 平凡算法  : 逐列访问 b[j*n+i]，步长=n，cache 不友好
 * 优化算法  : 外层按行、内层按列累加，步长=1，cache 友好
 *
 * 编译（由 run_exp1.bat 统一调用）：
 *   cl /O0 /Fe:build\exp1_O0.exe exp1.cpp    (MSVC)
 *   g++ -O0 -o build/exp1_O0 exp1.cpp        (MinGW/WSL)
 *
 * 用法：
 *   exp1_O2.exe <n> <runs> <opt_label>
 *   输出一行 CSV：opt_label,n,runs,t_naive_ms,t_opt_ms
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <windows.h>
#include <iomanip>
using namespace std;

// ── 数据初始化 ────────────────────────────────────
// 固定值，便于验证正确性；b 为行主序存储的 n×n 矩阵
void init_data(long long n, vector<double>& b, vector<double>& a) {
    for (long long i = 0; i < n; i++) {
        a[i] = static_cast<double>(i + 1);
        for (long long j = 0; j < n; j++)
            b[i * n + j] = static_cast<double>(i + j + 1);
    }
}

// ── 平凡算法：逐列访问 ────────────────────────────
// sum[i] = Σ_j b[j][i] * a[j]
// 内存访问：b[0*n+i], b[1*n+i], ... 步长=n，跨行，cache 不友好
void trivial_algorithm(long long n, const vector<double>& b,
                       const vector<double>& a, vector<double>& sum) {
    for (long long i = 0; i < n; i++) {
        sum[i] = 0.0;
        for (long long j = 0; j < n; j++)
            sum[i] += b[j * n + i] * a[j];
    }
}

// ── 优化算法：外层按行、内层按列累加 ──────────────
// 等价计算，但内层访问 b[j*n+i]（i连续），步长=1，cache 友好
void optimized_algorithm(long long n, const vector<double>& b,
                         const vector<double>& a, vector<double>& sum) {
    for (long long i = 0; i < n; i++) sum[i] = 0.0;
    for (long long j = 0; j < n; j++) {
        double aj = a[j];
        const double* row = &b[j * n];
        for (long long i = 0; i < n; i++)
            sum[i] += row[i] * aj;
    }
}

// ── 验证两种算法结果是否一致 ──────────────────────
bool verify(const vector<double>& r1, const vector<double>& r2, long long n) {
    for (long long i = 0; i < n; i++)
        if (fabs(r1[i] - r2[i]) > 1e-6 * (fabs(r1[i]) + 1.0)) return false;
    return true;
}

int main(int argc, char* argv[]) {
    SetConsoleOutputCP(65001);

    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <n> <runs> <opt_label>\n";
        return 1;
    }

    long long n   = atoll(argv[1]);
    int       runs = atoi(argv[2]);
    string    opt  = argv[3];

    vector<double> b(n * n), a(n), sum_naive(n), sum_opt(n);
    init_data(n, b, a);

    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

    // 预热（不计时）
    trivial_algorithm(n, b, a, sum_naive);
    optimized_algorithm(n, b, a, sum_opt);

    // 计时：平凡算法
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for (int r = 0; r < runs; r++) trivial_algorithm(n, b, a, sum_naive);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    double t_naive = (tail - head) * 1000.0 / freq;   // 总毫秒

    // 计时：优化算法
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for (int r = 0; r < runs; r++) optimized_algorithm(n, b, a, sum_opt);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    double t_opt = (tail - head) * 1000.0 / freq;

    bool ok = verify(sum_naive, sum_opt, n);

    // 输出 CSV 行：opt_label,n,runs,t_naive_ms(总),t_opt_ms(总),correct
    cout << fixed << setprecision(6)
         << opt << ","
         << n   << ","
         << runs << ","
         << t_naive << ","
         << t_opt   << ","
         << (ok ? "OK" : "FAIL")
         << "\n";

    return ok ? 0 : 1;
}