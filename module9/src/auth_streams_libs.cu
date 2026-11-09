/* ============================================================================
 * auth_streams_libs.cu
 * GPU-Accelerated Authentication Threat Detection project concurrent stream analyzer component with cuRAND Monte-Carlo testing and Thrust statistics
 *
 * Two LANL cyber1 datasets are scored concurrently on two CUDA streams:
 *   Dataset A: normal authentication events from auth.txt
 *   Dataset B: confirmed red-team compromise events from redteam.txt
 *              (ground truth of bad behavior -- smaller, and with a real
 *              attack signature rather than synthetic skew)
 * Both files are expected in the parent directory of this file by default.
 *
 *
 * Dataset: https://csr.lanl.gov/data/cyber1/
 * Build: nvcc -O2 -arch=sm_75 auth_streams_libs.cu -o auth_streams_libs -lcurand
 * Run:   ./auth_streams_libs --sizeA 1048576 --sizeB 65536 --threads 256 --blocks 4096 \
 *          --mc 16384 --sample 512
 * ==========================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <curand.h>
#include <curand_kernel.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/count.h>
#include <thrust/functional.h>

#define SCORE_LOOP 4096

/* ------------------------- command-line parameters ------------------------ */
static const char  *pathA    = "../auth.txt";     /* normal auth traffic   */
static const char  *pathB    = "../redteam.txt";  /* ground-truth attacks  */
static unsigned int sizeA    = 1024 * 1024;       /* max records from A    */
static unsigned int sizeB    = 64 * 1024;         /* max records from B    */
static unsigned int nthreads = 256;
static unsigned int nblocks  = 512;
static unsigned int mc_reps      = 16384;   /* Monte Carlo replicates       */
static unsigned int mc_sample    = 512;     /* random draws per replicate   */
static unsigned int mc_seed      = 12345;   /* fixed seed: reproducible run */

static void parse_args(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        if      (strcmp(argv[i], "--fileA") == 0 && i + 1 < argc)
            pathA = argv[++i];
        else if (strcmp(argv[i], "--fileB") == 0 && i + 1 < argc)
            pathB = argv[++i];
        else if (strcmp(argv[i], "--sizeA") == 0 && i + 1 < argc)
            sizeA = (unsigned int)atoi(argv[++i]);
        else if (strcmp(argv[i], "--sizeB") == 0 && i + 1 < argc)
            sizeB = (unsigned int)atoi(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
            nthreads = (unsigned int)atoi(argv[++i]);
        else if (strcmp(argv[i], "--blocks") == 0 && i + 1 < argc)
            nblocks = (unsigned int)atoi(argv[++i]);
        else if (strcmp(argv[i], "--mc") == 0 && i + 1 < argc)
            mc_reps = (unsigned int)atoi(argv[++i]);
        else if (strcmp(argv[i], "--sample") == 0 && i + 1 < argc)
            mc_sample = (unsigned int)atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
            mc_seed = (unsigned int)atoi(argv[++i]);
        else {
            fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
            fprintf(stderr, "Usage: %s [--fileA PATH] [--fileB PATH] [--sizeA N] "
                            "[--sizeB N] [--threads N] [--blocks N] "
                            "[--mc N] [--sample N] [--seed N]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }
    unsigned long long capacity = (unsigned long long)nblocks * nthreads;
    if (capacity < sizeA || capacity < sizeB) {
        fprintf(stderr, "Error: blocks*threads (%llu) must cover both dataset sizes "
                        "(%u, %u). Raise --blocks or lower --sizeA/--sizeB.\n",
                capacity, sizeA, sizeB);
        exit(EXIT_FAILURE);
    }
}

/* --------------------- LANL cyber1 file ingestion --------------------------*/
static void hash_record(const char *line, int is_redteam,
                        unsigned int *user_hash, unsigned int *ip_hash)
{
    unsigned int hu = 2166136261u, hi = 2166136261u;
    int field = 0;
    for (const char *p = line; *p && *p != '\n' && *p != '\r'; p++) {
        if (*p == ',') {
            field++;
            hu ^= ','; hu *= 16777619u;     /* mix separator into both hashes */
            hi ^= ','; hi *= 16777619u;
            continue;
        }
        if (field == 0)
            continue;                       /* skip the timestamp field */
        unsigned int *target = (is_redteam ? (field == 1) : (field <= 2)) ? &hu : &hi;
        *target ^= (unsigned char)*p;
        *target *= 16777619u;
    }
    *user_hash = hu;
    *ip_hash   = hi;
}

/* Read up to max_records lines; fills attempt[] and ip[]; returns count. */
static unsigned int read_dataset(const char *path, int is_redteam,
                                 unsigned int *attempt, unsigned int *ip,
                                 unsigned int max_records)
{
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Error: cannot open '%s'. LANL cyber1 auth.txt / redteam.txt "
                        "are expected in the parent directory by default.\n", path);
        exit(EXIT_FAILURE);
    }
    char line[1024];
    unsigned int n = 0;
    while (n < max_records && fgets(line, sizeof(line), fp)) {
        hash_record(line, is_redteam, &attempt[n], &ip[n]);
        n++;
    }
    fclose(fp);
    if (n == 0) {
        fprintf(stderr, "Error: no records read from '%s'\n", path);
        exit(EXIT_FAILURE);
    }
    return n;
}

/* --------------------------- scoring math --------------------------------- */
__device__ __forceinline__ unsigned int score_one(unsigned int a, unsigned int b)
{
    unsigned int s = 0;
    for (int i = 0; i < SCORE_LOOP; i++) {
        unsigned int diff = (a * 2654435761u) ^ (b * 40503u);
        s += diff * diff;
        a = a * 1664525u + 1013904223u;   /* LCG advance */
        b = b * 22695477u + 1u;
    }
    return s;
}

__global__ void score_attempts(const unsigned int *attempt_hash,
                               const unsigned int *ip_hash,
                               unsigned int *scores,
                               const unsigned int n)
{
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n)
        scores[tid] = score_one(attempt_hash[tid], ip_hash[tid]);
}

/* ------------------------- cuRAND Monte Carlo ----------------------------- */
__global__ void mc_init(unsigned int seed, curandState_t *states, unsigned int n)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < n)
        curand_init(seed,          /* seed can be shared by all cores       */
                    tid,           /* sequence number differs per thread    */
                    0,             /* no extra offset                       */
                    &states[tid]);
}

__global__ void mc_bootstrap(const curandState_t *states,
                             const unsigned int *pool_a,
                             const unsigned int *pool_b,
                             const unsigned int pool_size,
                             const unsigned int sample,
                             const double observed_mean,
                             unsigned int *extreme_count,
                             double *sim_sum,
                             const unsigned int mc)
{
    const unsigned int m = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (m >= mc)
        return;

    curandState_t state = states[m];       
    unsigned long long sum = 0;
    for (unsigned int j = 0; j < sample; j++) {
        unsigned int idx = curand(&state) % pool_size;  /* random resample */
        sum += score_one(pool_a[idx], pool_b[idx]);     /* full scoring math */
    }
    const double mean = (double)sum / sample;

    if (mean >= observed_mean)
        atomicAdd(extreme_count, 1u);
    atomicAdd(sim_sum, mean);
}

/* --------------------------- Thrust statistics ---------------------------- */
struct above_threshold {
    __host__ __device__ bool operator()(unsigned int x) const {
        return x > 0x80000000u;
    }
};

struct dataset_stats {
    double mean;
    unsigned int median;
    unsigned int flagged;
};

static dataset_stats gpu_stats(const unsigned int *host_scores, unsigned int n)
{
    thrust::host_vector<unsigned int> h(host_scores, host_scores + n);
    thrust::device_vector<unsigned int> d = h;      

    unsigned long long sum = thrust::reduce(d.begin(), d.end(),
                                            0ull, thrust::plus<unsigned long long>());
    unsigned int flagged = (unsigned int)thrust::count_if(d.begin(), d.end(),
                                                          above_threshold());
    thrust::device_vector<unsigned int> sorted = d; /* D2D copy, then sort   */
    thrust::sort(sorted.begin(), sorted.end());

    dataset_stats st;
    st.mean    = (double)sum / n;
    st.median  = sorted[n / 2];
    st.flagged = flagged;
    return st;  
}

/* Enqueue one full dataset (2 H2D copies, kernel, 1 D2H copy) on a stream */
static void run_dataset(cudaStream_t stream,
                        cudaEvent_t start_ev, cudaEvent_t stop_ev,
                        unsigned int *h_a, unsigned int *h_b, unsigned int *h_s,
                        unsigned int *d_a, unsigned int *d_b, unsigned int *d_s,
                        unsigned int n)
{
    cudaEventRecord(start_ev, stream);
    cudaMemcpyAsync(d_a, h_a, n * sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, h_b, n * sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
    score_attempts<<<nblocks, nthreads, 0, stream>>>(d_a, d_b, d_s, n);
    cudaMemcpyAsync(h_s, d_s, n * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    cudaEventRecord(stop_ev, stream);
}

int main(int argc, char **argv)
{
    parse_args(argc, argv);

    struct cudaDeviceProp prop;
    int whichDevice = 0;
    cudaGetDevice(&whichDevice);
    cudaGetDeviceProperties(&prop, whichDevice);

    printf("=== Authentication threat analysis: CUDA streams, events, cuRAND & Thrust ===\n");
    printf("GPU: %s\n", prop.name);
    printf("threads: %u | blocks: %u | MC replicates: %u | draws/replicate: %u\n\n",
           nthreads, nblocks, mc_reps, mc_sample);

    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    cudaEvent_t startA, stopA, startB, stopB;
    cudaEventCreate(&startA);
    cudaEventCreate(&stopA);
    cudaEventCreate(&startB);
    cudaEventCreate(&stopB);

    unsigned int *h_aA, *h_bA, *h_sA, *h_aB, *h_bB, *h_sB;
    cudaHostAlloc((void **)&h_aA, sizeA * sizeof(unsigned int), cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_bA, sizeA * sizeof(unsigned int), cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_sA, sizeA * sizeof(unsigned int), cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_aB, sizeB * sizeof(unsigned int), cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_bB, sizeB * sizeof(unsigned int), cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_sB, sizeB * sizeof(unsigned int), cudaHostAllocDefault);

    /* device memory */
    unsigned int *d_aA, *d_bA, *d_sA, *d_aB, *d_bB, *d_sB;
    cudaMalloc((void **)&d_aA, sizeA * sizeof(unsigned int));
    cudaMalloc((void **)&d_bA, sizeA * sizeof(unsigned int));
    cudaMalloc((void **)&d_sA, sizeA * sizeof(unsigned int));
    cudaMalloc((void **)&d_aB, sizeB * sizeof(unsigned int));
    cudaMalloc((void **)&d_bB, sizeB * sizeof(unsigned int));
    cudaMalloc((void **)&d_sB, sizeB * sizeof(unsigned int));

    /* load the real LANL cyber1 datasets */
    printf("Loading datasets ...\n");
    sizeA = read_dataset(pathA, 0, h_aA, h_bA, sizeA);
    sizeB = read_dataset(pathB, 1, h_aB, h_bB, sizeB);
    printf("Dataset A (normal):  %s -> %u records\n", pathA, sizeA);
    printf("Dataset B (attack):  %s -> %u records\n\n", pathB, sizeB);

    /* warmup execution of the kernel on stream0 */
    score_attempts<<<nblocks, nthreads, 0, stream0>>>(d_aA, d_bA, d_sA, sizeA);
    cudaStreamSynchronize(stream0);

    float tA = 0.0f, tB = 0.0f, tConcurrent = 0.0f, tSerial = 0.0f;

    /* --------------------- concurrent execution: 2 streams ----------------- */
    run_dataset(stream0, startA, stopA, h_aA, h_bA, h_sA, d_aA, d_bA, d_sA, sizeA);
    run_dataset(stream1, startB, stopB, h_aB, h_bB, h_sB, d_aB, d_bB, d_sB, sizeB);

    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);

    cudaEventElapsedTime(&tA, startA, stopA);
    cudaEventElapsedTime(&tB, startB, stopB);
    /* makespan: from start of A's work to end of B's work (captures overlap) */
    cudaEventElapsedTime(&tConcurrent, startA, stopB);

    /* ---------------- serialized baseline: both on stream0 ----------------- */
    run_dataset(stream0, startA, stopA, h_aA, h_bA, h_sA, d_aA, d_bA, d_sA, sizeA);
    run_dataset(stream0, startB, stopB, h_aB, h_bB, h_sB, d_aB, d_bB, d_sB, sizeB);

    cudaStreamSynchronize(stream0);

    cudaEventElapsedTime(&tSerial, startA, stopB);

    /* ------------------------- Thrust statistics --------------------------- */
    printf("Computing dataset statistics with Thrust ...\n");
    dataset_stats stA = gpu_stats(h_sA, sizeA);
    dataset_stats stB = gpu_stats(h_sB, sizeB);

    printf("\nConcurrent (2 streams):\n");
    printf("  Dataset A (normal):  %.2f ms\n", tA);
    printf("  Dataset B (attack):  %.2f ms\n", tB);
    printf("  Total makespan:      %.2f ms\n", tConcurrent);
    printf("Serialized (1 stream):\n");
    printf("  Total makespan:      %.2f ms\n", tSerial);
    printf("Overlap speedup:       %.2fx\n", tSerial / tConcurrent);

    printf("\nDetection metrics (Thrust: reduce/sort/count_if):\n");
    printf("  A (normal): mean = %.3e | median = %u | over threshold: %u (%.2f%%)\n",
           stA.mean, stA.median, stA.flagged, 100.0 * stA.flagged / sizeA);
    printf("  B (attack): mean = %.3e | median = %u | over threshold: %u (%.2f%%)\n",
           stB.mean, stB.median, stB.flagged, 100.0 * stB.flagged / sizeB);

    /* ------------------ cuRAND Monte Carlo bootstrap test ------------------ */
    printf("\nRunning cuRAND Monte Carlo bootstrap (%u replicates x %u draws) ...\n",
           mc_reps, mc_sample);

    const unsigned int pool_size = sizeA + sizeB;
    unsigned int *d_poolA = NULL, *d_poolB = NULL;
    curandState_t *d_states = NULL;
    unsigned int *d_extreme = NULL;
    double *d_simsum = NULL;
    cudaMalloc((void **)&d_poolA, pool_size * sizeof(unsigned int));
    cudaMalloc((void **)&d_poolB, pool_size * sizeof(unsigned int));
    cudaMalloc((void **)&d_states, mc_reps * sizeof(curandState_t));
    cudaMalloc((void **)&d_extreme, sizeof(unsigned int));
    cudaMalloc((void **)&d_simsum, sizeof(double));

    /* pool = dataset A hashes followed by dataset B hashes (both on device) */
    cudaMemcpy(d_poolA,        d_aA, sizeA * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_poolA + sizeA, d_aB, sizeB * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_poolB,        d_bA, sizeA * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_poolB + sizeA, d_bB, sizeB * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    cudaMemset(d_extreme, 0, sizeof(unsigned int));
    cudaMemset(d_simsum,  0, sizeof(double));

    const unsigned int mc_blocks = (mc_reps + nthreads - 1) / nthreads;

    cudaEvent_t mcStart, mcStop;
    cudaEventCreate(&mcStart);
    cudaEventCreate(&mcStop);
    float tMC = 0.0f;

    cudaEventRecord(mcStart, 0);
    mc_init<<<mc_blocks, nthreads>>>(mc_seed, d_states, mc_reps);
    mc_bootstrap<<<mc_blocks, nthreads>>>(d_states, d_poolA, d_poolB, pool_size,
                                          mc_sample, stB.mean, d_extreme, d_simsum,
                                          mc_reps);
    cudaEventRecord(mcStop, 0);
    cudaEventSynchronize(mcStop);
    cudaEventElapsedTime(&tMC, mcStart, mcStop);

    unsigned int extreme = 0;
    double sim_sum = 0.0;
    cudaMemcpy(&extreme, d_extreme, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sim_sum, d_simsum, sizeof(double), cudaMemcpyDeviceToHost);

    const double p_value = ((double)extreme + 1.0) / ((double)mc_reps + 1.0);
    printf("  Monte Carlo time:    %.2f ms\n", tMC);
    printf("  Observed B mean:     %.3e\n", stB.mean);
    printf("  Simulated null mean: %.3e  (avg over %u replicates)\n",
           sim_sum / mc_reps, mc_reps);
    printf("  Replicates >= B:     %u of %u\n", extreme, mc_reps);
    printf("  Bootstrap p-value:   %.5f  -> dataset B is %s\n", p_value,
           p_value < 0.05 ? "Significantly more anomalous than pooled traffic (reject null at 0.05)"
                          : "not distinguishable from pooled traffic at 0.05");

    cudaFree(d_poolA); cudaFree(d_poolB);
    cudaFree(d_states); cudaFree(d_extreme); cudaFree(d_simsum);
    cudaEventDestroy(mcStart); cudaEventDestroy(mcStop);

    cudaFreeHost(h_aA); cudaFreeHost(h_bA); cudaFreeHost(h_sA);
    cudaFreeHost(h_aB); cudaFreeHost(h_bB); cudaFreeHost(h_sB);
    cudaFree(d_aA); cudaFree(d_bA); cudaFree(d_sA);
    cudaFree(d_aB); cudaFree(d_bB); cudaFree(d_sB);
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
    cudaEventDestroy(startA); cudaEventDestroy(stopA);
    cudaEventDestroy(startB); cudaEventDestroy(stopB);
    cudaDeviceReset();
    return EXIT_SUCCESS;
}
