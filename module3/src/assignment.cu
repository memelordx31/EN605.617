#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cuda_runtime.h>

#define N 2560 

__global__ void multiply(float* a, float* b, float* c, int n) {
    int total = n * n;
    int stride = blockDim.x * gridDim.x;

    int lane = threadIdx.x % warpSize;
    bool upperHalf = (lane >= warpSize / 2);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride) {
        int row = i / n;
        int col = i % n;
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            float x = a[row * n + k] * b[k * n + col];
            if (upperHalf)
                sum += x * x;
            else
                sum += sqrtf(x);
        }
        c[i] = sum;
    }
}

void multiplyHost(float* a, float* b, float* c, int n) {
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                float x = a[row * n + k] * b[k * n + col];
                if (x > 0.25f)
                    sum += x * x;
                else
                    sum += sqrtf(x);
            }
            c[row * n + col] = sum;
        }
    }
}

static double mean(const std::vector<long long>& v) {
    long long s = 0;
    for (auto x : v) s += x;
    return double(s) / v.size();
}

static double stddev(const std::vector<long long>& v, double m) {
    double s = 0;
    for (auto x : v) {
        double d = x - m;
        s += d * d;
    }
    return std::sqrt(s / v.size());
}

static long long median(std::vector<long long> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n % 2 == 1) return v[n / 2];
    return (v[n / 2 - 1] + v[n / 2]) / 2;
}

int main(int argc, char** argv)
{
    int totalThreads = (1 << 20);
    int blockSize = 256;
    int iterations = 10;   // number of timing runs

    if (argc >= 2) totalThreads = atoi(argv[1]);
    if (argc >= 3) blockSize = atoi(argv[2]);
    if (argc >= 4) iterations = atoi(argv[3]);

    int numBlocks = totalThreads / blockSize;
    if (totalThreads % blockSize != 0) {
        ++numBlocks;
        totalThreads = numBlocks * blockSize;
        std::cout << "Warning: totalThreads rounded to " << totalThreads << "\n";
    }

    size_t size = size_t(N) * N * sizeof(float);
    float* a = (float*)malloc(size);
    float* b = (float*)malloc(size);
    float* c = (float*)malloc(size);
    float* cHost = (float*)malloc(size);

    float* dev_a, * dev_b, * dev_c;
    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_c, size);

    srand(42);
    for (size_t i = 0; i < size / sizeof(float); i++) {
        a[i] = float(rand()) / RAND_MAX;
        b[i] = float(rand()) / RAND_MAX;
    }

    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

    std::vector<long long> gpuTimes;
    std::vector<long long> cpuTimes;

    for (int it = 0; it < iterations; it++) {
        auto start = std::chrono::high_resolution_clock::now();
        multiply << <numBlocks, blockSize >> > (dev_a, dev_b, dev_c, N);
        cudaDeviceSynchronize();
        auto stop = std::chrono::high_resolution_clock::now();

        long long gpuNs = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        gpuTimes.push_back(gpuNs);

        cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

        auto startHost = std::chrono::high_resolution_clock::now();
        multiplyHost(a, b, cHost, N);
        auto stopHost = std::chrono::high_resolution_clock::now();

        long long cpuNs = std::chrono::duration_cast<std::chrono::nanoseconds>(stopHost - startHost).count();
        cpuTimes.push_back(cpuNs);

        std::cout << "Iteration " << it + 1 << ": GPU " << gpuNs << " ns, CPU " << cpuNs << " ns\n";
    }

    double gpuMean = mean(gpuTimes);
    double cpuMean = mean(cpuTimes);
    double gpuStd = stddev(gpuTimes, gpuMean);
    double cpuStd = stddev(cpuTimes, cpuMean);
    long long gpuMin = *std::min_element(gpuTimes.begin(), gpuTimes.end());
    long long gpuMax = *std::max_element(gpuTimes.begin(), gpuTimes.end());
    long long cpuMin = *std::min_element(cpuTimes.begin(), cpuTimes.end());
    long long cpuMax = *std::max_element(cpuTimes.begin(), cpuTimes.end());
    long long gpuMedian = median(gpuTimes);
    long long cpuMedian = median(cpuTimes);

    std::cout << "\n GPU Statistics (ns) \n";
    std::cout << "Mean:   " << gpuMean << "\n";
    std::cout << "StdDev: " << gpuStd << "\n";
    std::cout << "Min:    " << gpuMin << "\n";
    std::cout << "Max:    " << gpuMax << "\n";
    std::cout << "Median: " << gpuMedian << "\n";

    std::cout << "\n CPU Statistics (ns) \n";
    std::cout << "Mean:   " << cpuMean << "\n";
    std::cout << "StdDev: " << cpuStd << "\n";
    std::cout << "Min:    " << cpuMin << "\n";
    std::cout << "Max:    " << cpuMax << "\n";
    std::cout << "Median: " << cpuMedian << "\n";

    std::ofstream csv("statistics.csv");
    csv << "iteration,gpu_ns,cpu_ns\n";
    for (int i = 0; i < iterations; i++) {
        csv << (i + 1) << "," << gpuTimes[i] << "," << cpuTimes[i] << "\n";
    }
    csv << "\n";
    csv << "stat,gpu_ns,cpu_ns\n";
    csv << "mean," << gpuMean << "," << cpuMean << "\n";
    csv << "stddev," << gpuStd << "," << cpuStd << "\n";
    csv << "min," << gpuMin << "," << cpuMin << "\n";
    csv << "max," << gpuMax << "," << cpuMax << "\n";
    csv << "median," << gpuMedian << "," << cpuMedian << "\n";
    csv.close();

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(a); free(b); free(c); free(cHost);

    return 0;
}
