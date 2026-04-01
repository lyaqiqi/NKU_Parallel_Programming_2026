#include <iostream>
#include <vector>
#include <windows.h>
#include <iomanip>

using namespace std;

// 初始化测试数据
void init_data(long long n, vector<double>& b, vector<double>& a) {
    if (n == 0) return;
    for (long long i = 0; i < n; i++) {
        a[i] = i * 1.0;
        for (long long j = 0; j < n; j++) {
            b[i * n + j] = i + j; 
        }
    }
}

// 平凡算法 (NormalAlg)
void trivial_algorithm(long long n, const vector<double>& b, const vector<double>& a, vector<double>& sum) {
    if (n == 0) return;
    for (long long i = 0; i < n; i++) {
        sum[i] = 0.0;
        for (long long j = 0; j < n; j++) {
            sum[i] += b[j * n + i] * a[j];
        }
    }
}

// Cache优化算法 (CacheOptAlg)
void optimized_algorithm(long long n, const vector<double>& b, const vector<double>& a, vector<double>& sum) {
    if (n == 0) return;
    for (long long i = 0; i < n; i++) {
        sum[i] = 0.0;
    }
    for (long long j = 0; j < n; j++) {
        for (long long i = 0; i < n; i++) {
            sum[i] += b[j * n + i] * a[j];
        }
    }
}

// 循环展开优化算法 (UnrollAlg, 步长为4)
void unroll_algorithm(long long n, const vector<double>& b, const vector<double>& a, vector<double>& sum) {
    if (n == 0) return;
    for (long long i = 0; i < n; i++) {
        sum[i] = 0.0;
    }
    for (long long j = 0; j < n; j++) {
        long long i = 0;
        // 每次处理4个元素
        for (; i <= n - 4; i += 4) {
            sum[i]     += b[j * n + i] * a[j];
            sum[i + 1] += b[j * n + (i + 1)] * a[j];
            sum[i + 2] += b[j * n + (i + 2)] * a[j];
            sum[i + 3] += b[j * n + (i + 3)] * a[j];
        }
        // 处理末尾不足4个的剩余元素
        for (; i < n; i++) {
            sum[i] += b[j * n + i] * a[j];
        }
    }
}

struct TestConfig {
    long long n;
    int runs;
};

int main() {
    SetConsoleOutputCP(65001); // 强制 UTF-8 避免乱码
    
    vector<TestConfig> configs = {
        {0, 10000}, {10, 10000}, {20, 10000}, {50, 5000},
        {100, 1000}, {150, 1000}, {200, 500}, {500, 100},
        {1000, 10}, {2000, 10}, {3000, 1}, {5000, 1},
        {10000, 1}, {20000, 1}
    };

    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

    // 打印表头 (加宽以容纳三列)
    cout << setfill('-') << setw(140) << "-" << setfill(' ') << endl;
    cout << left 
         << setw(12) << "规模\\时间" 
         << setw(10) << "次数" 
         << setw(18) << "总 NormalAlg" 
         << setw(18) << "总 CacheOpt" 
         << setw(18) << "总 UnrollAlg" 
         << setw(18) << "平均 NormalAlg" 
         << setw(18) << "平均 CacheOpt" 
         << "平均 UnrollAlg" << endl;
    cout << setfill('-') << setw(140) << "-" << setfill(' ') << endl;

    for (const auto& config : configs) {
        long long n = config.n;
        int runs = config.runs;

        vector<double> b(n * n);
        vector<double> a(n);
        vector<double> sum_normal(n), sum_opt(n), sum_unroll(n);

        init_data(n, b, a);

        // 1. 测试 NormalAlg
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        for (int r = 0; r < runs; r++) trivial_algorithm(n, b, a, sum_normal);
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        double total_normal = (tail - head) * 1000.0 / freq;

        // 2. 测试 CacheOptAlg
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        for (int r = 0; r < runs; r++) optimized_algorithm(n, b, a, sum_opt);
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        double total_opt = (tail - head) * 1000.0 / freq;

        // 3. 测试 UnrollAlg
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        for (int r = 0; r < runs; r++) unroll_algorithm(n, b, a, sum_unroll);
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        double total_unroll = (tail - head) * 1000.0 / freq;

        // 格式化输出
        cout << left 
             << setw(12) << n 
             << setw(10) << runs 
             << fixed << setprecision(4)
             << setw(18) << total_normal 
             << setw(18) << total_opt 
             << setw(18) << total_unroll
             << setprecision(5)
             << setw(18) << total_normal / runs 
             << setw(18) << total_opt / runs 
             << total_unroll / runs << endl;
             
        // 释放内存
        b.clear(); b.shrink_to_fit();
        a.clear(); a.shrink_to_fit();
        sum_normal.clear(); sum_normal.shrink_to_fit();
        sum_opt.clear(); sum_opt.shrink_to_fit();
        sum_unroll.clear(); sum_unroll.shrink_to_fit();
    }
    
    cout << setfill('-') << setw(140) << "-" << setfill(' ') << endl;
    return 0;
}