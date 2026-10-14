// gpu_arch.cu
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Use in nvcc as: -arch=sm_%d%d\n", prop.major, prop.minor);
    return 0;
}
