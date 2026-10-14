/* ============================================================================
 * auth_streams.cu 
 * GPU-Accelerated Authentication Threat Detection project concurrent stream analyzer component
 *
 * Two LANL cyber1 datasets are scored concurrently on two CUDA streams:
 *   Dataset A: normal authentication events from auth.txt
 *   Dataset B: confirmed red-team compromise events from redteam.txt
 *              (ground truth of bad behavior -- smaller, and with a real
 *              attack signature rather than synthetic skew)
 * Both files are expected in the parent directory of this file by default.
 *
 * The same scoring kernel is executed once per dataset on a different
 * stream, with events measuring each stream's latency plus the overall
 * makespan, and a serialized baseline run on a single stream for an
 * explicit overlap comparison.
 *
 * Dataset: https://csr.lanl.gov/data/cyber1/  
 * Build: nvcc -O2 -arch=sm_75 auth_streams.cu -o auth_streams
 * Run:   ./auth_streams --sizeA 131072 --sizeB 65536 --threads 256 --blocks 512
 * ==========================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SCORE_LOOP 4096

/* ------------------------- command-line parameters ------------------------ */
static const char  *pathA    = "../auth.txt";     /* normal auth traffic   */
static const char  *pathB    = "../redteam.txt";  /* ground-truth attacks  */
static unsigned int sizeA    = 1024 * 1024;       /* max records from A    */
static unsigned int sizeB    = 64 * 1024;         /* max records from B    */
static unsigned int nthreads = 256;
static unsigned int nblocks  = 512;

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
        else {
            fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
            fprintf(stderr, "Usage: %s [--fileA PATH] [--fileB PATH] [--sizeA N] "
                            "[--sizeB N] [--threads N] [--blocks N]\n", argv[0]);
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

/* --------------------- LANL cyber1 file ingestion -------------------------- */
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

/* --------------------------- scoring kernel ------------------------------- */
__global__ void score_attempts(const unsigned int *attempt_hash,
                               const unsigned int *ip_hash,
                               unsigned int *scores,
                               const unsigned int n)
{
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        unsigned int a = attempt_hash[tid];   /* register variables */
        unsigned int b = ip_hash[tid];
        unsigned int s = 0;
        for (int i = 0; i < SCORE_LOOP; i++) {
            unsigned int diff = (a * 2654435761u) ^ (b * 40503u);
            s += diff * diff;
            a = a * 1664525u + 1013904223u;   /* LCG advance */
            b = b * 22695477u + 1u;
        }
        scores[tid] = s;
    }
}

static double dataset_mean(const unsigned int *scores, unsigned int n,
                           unsigned int *over_thresh)
{
    double sum = 0.0;
    *over_thresh = 0;
    for (unsigned int i = 0; i < n; i++) {
        sum += scores[i];
        if (scores[i] > 0x80000000u)
            (*over_thresh)++;
    }
    return sum / n;
}

/* Enqueue one full dataset (2 H2D copies, kernel, 1 D2H copy) on a stream,
 * bracketed by events recorded ON that stream. */
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

    printf("=== Authentication threat analysis: CUDA streams & events ===\n");
    printf("GPU: %s\n", prop.name);
    printf("threads: %u | blocks: %u\n\n", nthreads, nblocks);

    /* streams and events */
    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    cudaEvent_t startA, stopA, startB, stopB;
    cudaEventCreate(&startA);
    cudaEventCreate(&stopA);
    cudaEventCreate(&startB);
    cudaEventCreate(&stopB);

    /* pinned host memory for both datasets */
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

    /* load the real LANL cyber1 datasets (records are FNV-1a hashed host-side) */
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

    /* ------------------------------ metrics -------------------------------- */
    unsigned int flaggedA, flaggedB;
    double meanA = dataset_mean(h_sA, sizeA, &flaggedA);
    double meanB = dataset_mean(h_sB, sizeB, &flaggedB);

    printf("Concurrent (2 streams):\n");
    printf("  Dataset A (normal):  %.2f ms\n", tA);
    printf("  Dataset B (attack):  %.2f ms\n", tB);
    printf("  Total makespan:      %.2f ms\n", tConcurrent);
    printf("Serialized (1 stream):\n");
    printf("  Total makespan:      %.2f ms\n", tSerial);
    printf("Overlap speedup:       %.2fx\n", tSerial / tConcurrent);

    printf("\nDetection metrics per dataset:\n");
    printf("  A (normal): mean score = %.3e | over threshold: %u (%.2f%%)\n",
           meanA, flaggedA, 100.0 * flaggedA / sizeA);
    printf("  B (attack): mean score = %.3e | over threshold: %u (%.2f%%)\n",
           meanB, flaggedB, 100.0 * flaggedB / sizeB);

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
