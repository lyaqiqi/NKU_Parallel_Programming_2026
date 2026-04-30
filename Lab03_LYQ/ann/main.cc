#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include <omp.h>
#include "hnswlib/hnswlib/hnswlib.h"
#include "flat_scan.h"
#include "simd_search.h"
#include "sq_search.h"
#include "pq_search.h"
// 可以自行添加需要的头文件

using namespace hnswlib;

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();

    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";

    return data;
}

struct SearchResult
{
    float recall;
    int64_t latency; // 单位us
};

void build_index(float* base, size_t base_number, size_t vecdim)
{
    const int efConstruction = 150; // 为防止索引构建时间过长，efc建议设置200以下
    const int M = 16; // M建议设置为16以下

    HierarchicalNSW<float> *appr_alg;
    InnerProductSpace ipspace(vecdim);
    appr_alg = new HierarchicalNSW<float>(&ipspace, base_number, M, efConstruction);

    appr_alg->addPoint(base, 0);
    #pragma omp parallel for
    for(int i = 1; i < base_number; ++i) {
        appr_alg->addPoint(base + 1ll*vecdim*i, i);
    }

    char path_index[1024] = "files/hnsw.index";
    appr_alg->saveIndex(path_index);
}


int main(int argc, char *argv[])
{
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "/anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    // 只测试前2000条查询
    test_number = 2000;

    const size_t k = 10;

    std::vector<SearchResult> results;
    results.resize(test_number);

    // 如果你需要保存索引，可以在这里添加你需要的函数，你可以将下面的注释删除来查看pbs是否将build.index返回到你的files目录中
    // 要保存的目录必须是files/*
    // 每个人的目录空间有限，不需要的索引请及时删除，避免占空间太大
    // 不建议在正式测试查询时同时构建索引，否则性能波动会较大
    // 下面是一个构建hnsw索引的示例
    // build_index(base, base_number, vecdim);

    // PQ 索引：从 Python 预生成的码本文件加载（量化开销不计入查询延迟）
    PQIndex pq_idx;
    bool pq_ok = pq_idx.load("files/pq_codebook.bin", "files/pq_codes.bin",
                              base, base_number, vecdim);
    if (!pq_ok) {
        std::cerr << "PQIndex load failed — run generate_pq.py first\n";
        return 1;
    }

    // p-sweep：1..50 逐个取，之后跳跃取
    std::vector<size_t> p_list;
    for (size_t p = 1; p <= 50; ++p) p_list.push_back(p);
    {
        const size_t extra[] = {75, 100, 150, 200, 300, 500, 750,
                                 1000, 1500, 2000, 3000, 5000};
        for (size_t p : extra) p_list.push_back(p);
    }

    float last_recall = 0, last_latency_us = 0;
    for (size_t pq_p : p_list) {

        // 查询测试代码
        for(int i = 0; i < (int)test_number; ++i) {
            const unsigned long Converter = 1000 * 1000;
            struct timeval val;
            gettimeofday(&val, NULL);

            // 该文件已有代码中你只能修改该函数的调用方式
            // 可以任意修改函数名，函数参数或者改为调用成员函数，但是不能修改函数返回值。
            auto res = pq_idx.search(test_query + i*vecdim, k, pq_p);

            struct timeval newVal;
            gettimeofday(&newVal, NULL);
            int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec)
                         - (val.tv_sec   * Converter + val.tv_usec);

            std::set<uint32_t> gtset;
            for(int j = 0; j < (int)k; ++j){
                int t = test_gt[j + i*test_gt_d];
                gtset.insert(t);
            }

            size_t acc = 0;
            while (res.size()) {
                int x = res.top().second;
                if(gtset.find(x) != gtset.end()) ++acc;
                res.pop();
            }
            results[i] = {(float)acc / k, diff};
        }

        float avg_recall = 0, avg_latency_us = 0;
        for(int i = 0; i < (int)test_number; ++i) {
            avg_recall     += results[i].recall;
            avg_latency_us += results[i].latency;
        }
        avg_recall     /= test_number;
        avg_latency_us /= test_number;

        // output: p=X latency=Yms recall=Z
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "p=" << pq_p
                  << " latency=" << avg_latency_us / 1000.0f << "ms"
                  << " recall="  << avg_recall << "\n";
        std::cout.flush();

        last_recall     = avg_recall;
        last_latency_us = avg_latency_us;
    }

    // 最后一个 p 值的结果保持标准格式
    std::cout << "average recall: "       << last_recall     << "\n";
    std::cout << "average latency (us): " << last_latency_us << "\n";
    return 0;
}
