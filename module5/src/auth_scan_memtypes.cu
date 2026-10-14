/* ============================================================================
 * auth_scan_memtypes.cu
 * GPU-Accelerated Authentication Threat Detection project memory-type scanner component
 *
 * Reads hashed login attempts from the LANL cyber1 auth.txt dataset
 * (expected in the parent directory of this file by default) and scans
 * them for anomalies. The same anomaly-scoring operation is performed by
 * four kernel variants so the timing of the different CUDA memory types
 * can be compared:
 *
 *   scan_literal : masks hard-coded as literals      
 *   scan_const   : masks read from __constant__ mem  
 *   scan_gmem    : masks read from __device__ globals
 *   scan_shared  : const masks + __shared__ per-block threat reduction
 * Dataset: https://csr.lanl.gov/data/cyber1/
 * Build: nvcc -O2 -arch=sm_75 auth_scan_memtypes.cu -o auth_scan_memtypes
 * Run:   ./auth_scan_memtypes --elements 131072 --threads 256 --blocks 512 --runs 6
 *        ./auth_scan_memtypes --file ../auth.txt --elements 1000000
 * ==========================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KERNEL_LOOP 4096
#define THREAT_BIT  0x40000000u   /* flag bit in the final anomaly score */

static const char  *auth_path    = "../auth.txt";  /* LANL cyber1 auth log  */
static unsigned int num_elements = 131072;   /* max auth records to read    */
static unsigned int num_threads  = 256;
static unsigned int num_blocks   = 512;
static unsigned int num_runs     = 6;

static void parse_args(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        if      (strcmp(argv[i], "--file") == 0 && i + 1 < argc)
            auth_path = argv[++i];
        else if (strcmp(argv[i], "--elements") == 0 && i + 1 < argc)
            num_elements = (unsigned int)atoi(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
            num_threads = (unsigned int)atoi(argv[++i]);
        else if (strcmp(argv[i], "--blocks") == 0 && i + 1 < argc)
            num_blocks = (unsigned int)atoi(argv[++i]);
        else if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc)
            num_runs = (unsigned int)atoi(argv[++i]);
        else {
            fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
            fprintf(stderr, "Usage: %s [--file PATH] [--elements N] [--threads N] "
                            "[--blocks N] [--runs N]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }
}

/* --------------------- LANL cyber1 auth.txt ingestion ----------------------*/
static unsigned int hash_auth_record(const char *line)
{
    unsigned int h = 2166136261u;
    int field = 0;
    for (const char *p = line; *p && *p != '\n' && *p != '\r'; p++) {
        if (*p == ',') {
            field++;
            h ^= ',';
            h *= 16777619u;
            continue;
        }
        if (field == 0)
            continue;                       /* skip the timestamp field */
        h ^= (unsigned char)*p;
        h *= 16777619u;
    }
    return h;
}

/* Read up to max_records lines from the auth log into out[]; returns the
 * number of records actually read (file may contain fewer). */
static unsigned int read_auth_records(const char *path, unsigned int *out,
                                      unsigned int max_records)
{
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Error: cannot open '%s'. The LANL cyber1 auth.txt is "
                        "expected in the parent directory by default.\n", path);
        exit(EXIT_FAILURE);
    }
    char line[1024];
    unsigned int n = 0;
    while (n < max_records && fgets(line, sizeof(line), fp))
        out[n++] = hash_auth_record(line);
    fclose(fp);
    if (n == 0) {
        fprintf(stderr, "Error: no records read from '%s'\n", path);
        exit(EXIT_FAILURE);
    }
    return n;
}

/* -------------------- constant memory: known-bad hash masks ----------------- */
__constant__ static const unsigned int const_breach_mask = 0x55555555u; /* breached password hash */
__constant__ static const unsigned int const_ip_mask     = 0x77777777u; /* suspicious source ip */
__constant__ static const unsigned int const_repeat_mask = 0x33333333u; /* repeated attempts */
__constant__ static const unsigned int const_admin_mask  = 0x11111111u; /* admin account probe */

__device__ static unsigned int gmem_breach_mask = 0x55555555u;
__device__ static unsigned int gmem_ip_mask     = 0x77777777u;
__device__ static unsigned int gmem_repeat_mask = 0x33333333u;
__device__ static unsigned int gmem_admin_mask  = 0x11111111u;

/* ------------------------------ kernel variants --------------------------- */
__global__ void scan_literal(unsigned int * const data, const unsigned int num_elements)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < num_elements) {
        unsigned int d       = data[tid];   
        const unsigned int attempt = data[tid];
        for (int i = 0; i < KERNEL_LOOP; i++) {
            d ^= 0x55555555u;               /* known-bad masks as literals */
            d |= 0x77777777u;
            d &= 0x33333333u;
            d |= 0x11111111u;
            d += attempt;                  
        }
        data[tid] = d;
    }
}

__global__ void scan_const(unsigned int * const data, const unsigned int num_elements)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < num_elements) {
        unsigned int d       = data[tid];   
        const unsigned int attempt = data[tid];
        for (int i = 0; i < KERNEL_LOOP; i++) {
            d ^= const_breach_mask;         /* known-bad masks from __constant__ */
            d |= const_ip_mask;
            d &= const_repeat_mask;
            d |= const_admin_mask;
            d += attempt;
        }
        data[tid] = d;
    }
}

__global__ void scan_gmem(unsigned int * const data, const unsigned int num_elements)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < num_elements) {
        unsigned int d       = data[tid];   
        const unsigned int attempt = data[tid];
        for (int i = 0; i < KERNEL_LOOP; i++) {
            d ^= gmem_breach_mask;          /* known-bad masks from global memory */
            d |= gmem_ip_mask;
            d &= gmem_repeat_mask;
            d |= gmem_admin_mask;
            d += attempt;
        }
        data[tid] = d;
    }
}

__global__ void scan_shared(unsigned int * const data, unsigned int * const threat_count,
                            const unsigned int num_elements)
{
    __shared__ unsigned int block_threats;  /* shared memory: per-block threat counter */
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (threadIdx.x == 0)
        block_threats = 0;
    __syncthreads();

    if (tid < num_elements) {
        unsigned int d       = data[tid];   
        const unsigned int attempt = data[tid];
        for (int i = 0; i < KERNEL_LOOP; i++) {
            d ^= const_breach_mask;
            d |= const_ip_mask;
            d &= const_repeat_mask;
            d |= const_admin_mask;
            d += attempt;
        }
        data[tid] = d;
        if (d & THREAT_BIT)
            atomicAdd(&block_threats, 1u);
    }
    __syncthreads();

    if (threadIdx.x == 0 && block_threats > 0)
        atomicAdd(threat_count, block_threats);
}

/* ------------------------------- host driver ------------------------------ */
static const char *variant_name[4] = { "Literal", "Constant", "Global  ", "Shared  " };

static void launch_variant(int variant, unsigned int *d_data, unsigned int *d_threats)
{
    switch (variant) {
    case 0: scan_literal<<<num_blocks, num_threads>>>(d_data, num_elements); break;
    case 1: scan_const  <<<num_blocks, num_threads>>>(d_data, num_elements); break;
    case 2: scan_gmem   <<<num_blocks, num_threads>>>(d_data, num_elements); break;
    case 3: scan_shared <<<num_blocks, num_threads>>>(d_data, d_threats, num_elements); break;
    }
}

int main(int argc, char **argv)
{
    parse_args(argc, argv);

    struct cudaDeviceProp device_prop;
    int device_num = 0;
    cudaGetDevice(&device_num);
    cudaGetDeviceProperties(&device_prop, device_num);

    printf("=== Authentication threat scan: CUDA memory-type timing comparison ===\n");
    printf("GPU: %s | dataset: %s | threads: %u | blocks: %u | runs: %u\n\n",
           device_prop.name, auth_path, num_threads, num_blocks, num_runs);

    /* host memory: array of hashed login attempts */
    unsigned int *host_attempts = (unsigned int *)malloc(num_elements * sizeof(unsigned int));
    unsigned int *host_output   = (unsigned int *)malloc(num_elements * sizeof(unsigned int));
    unsigned int *ref_output    = (unsigned int *)malloc(num_elements * sizeof(unsigned int));
    if (!host_attempts || !host_output || !ref_output) {
        fprintf(stderr, "host malloc failed\n");
        return EXIT_FAILURE;
    }

    printf("Reading up to %u authentication records from %s ...\n",
           num_elements, auth_path);
    num_elements = read_auth_records(auth_path, host_attempts, num_elements);
    printf("Read %u records.\n", num_elements);

    if ((unsigned long long)num_blocks * num_threads < num_elements) {
        fprintf(stderr, "Error: blocks*threads (%u*%u=%llu) < records read (%u).\n"
                        "Raise --blocks or --threads so every record gets a thread.\n",
                num_blocks, num_threads,
                (unsigned long long)num_blocks * num_threads, num_elements);
        return EXIT_FAILURE;
    }

    const unsigned int num_bytes = num_elements * sizeof(unsigned int);

    /* global memory */
    unsigned int *d_data    = NULL;
    unsigned int *d_threats = NULL;
    cudaMalloc(&d_data, num_bytes);
    cudaMalloc(&d_threats, sizeof(unsigned int));

    float total_time[4] = {0}, best[4] = {0};
    unsigned int threats = 0;
    int all_match = 1;

    for (int v = 0; v < 4; v++) {
        /* fresh input for this variant */
        cudaMemcpy(d_data, host_attempts, num_bytes, cudaMemcpyHostToDevice);
        cudaMemset(d_threats, 0, sizeof(unsigned int));

        /* warmup launch, as in constant_memory.cu */
        launch_variant(v, d_data, d_threats);
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreateWithFlags(&stop, cudaEventBlockingSync);

        for (unsigned int run = 0; run < num_runs; run++) {
            cudaMemset(d_threats, 0, sizeof(unsigned int));

            cudaEventRecord(start, 0);
            launch_variant(v, d_data, d_threats);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);

            float delta = 0.0f;
            cudaEventElapsedTime(&delta, start, stop);
            total_time[v] += delta;
            if (run == 0 || delta < best[v])
                best[v] = delta;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaMemcpy(host_output, d_data, num_bytes, cudaMemcpyDeviceToHost);
        if (v == 0) {
            memcpy(ref_output, host_output, num_bytes);
        } else if (memcmp(ref_output, host_output, num_bytes) != 0) {
            all_match = 0;
        }
        if (v == 3)
            cudaMemcpy(&threats, d_threats, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    }

    printf("\nVariant  | avg time (ms) | best time (ms) | memory types used\n");
    printf("---------+---------------+----------------+----------------------------\n");
    for (int v = 0; v < 4; v++) {
        printf("%s | %13.2f | %14.2f | %s\n", variant_name[v],
               (double)total_time[v] / num_runs, (double)best[v],
               v == 0 ? "host + global + registers (literal masks)" :
               v == 1 ? "host + global + registers + __constant__" :
               v == 2 ? "host + global + registers (__device__ globals)" :
                        "host + global + registers + __constant__ + __shared__");
    }
    printf("\nThreats detected (shared-memory reduction): %u of %u attempts (%.2f%%)\n",
           threats, num_elements, 100.0 * threats / num_elements);
    printf("Verification: all 4 variants produced identical output: %s\n",
           all_match ? "YES" : "NO");

    int fastest = 0;
    for (int v = 1; v < 4; v++)
        if (total_time[v] < total_time[fastest]) fastest = v;
    printf("Fastest variant: %s\n", variant_name[fastest]);

    free(host_attempts);
    free(host_output);
    free(ref_output);
    cudaFree(d_data);
    cudaFree(d_threats);
    cudaDeviceReset();
    return EXIT_SUCCESS;
}
