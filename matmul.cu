#include <cstdio>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0) 

#define HOST_CHECK(ptr) do { \
    if ((ptr) == nullptr) { \
        printf("host allocation failed at %s:%d\n", __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0) // Added missing parenthesis here

// definitions
#define Nc 1000000
#define m 1024
#define k 1024
#define n 1024

// fill row-major matrix with random floats [-1, 1]
void fill_random(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
    }
}

// visualizer: prints the top-left block of a matrix (made w ai)
void visualize_matrix(const char* name, const float* mat, int rows, int cols, int block_size = 8) {
    int r = (rows < block_size) ? rows : block_size;
    int c = (cols < block_size) ? cols : block_size;

    printf("\n=== %s (Top-Left %dx%d) ===\n", name, r, c);
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            printf("%8.3f ", mat[i * cols + j]);
        }
        printf("\n");
    }
    printf("==================================\n");
}

__global__ void matmul_kernel(float* A, float* B, float* C, int M, int K, int N) {
    int x = blockDim.x * blockIdx.x + threadIdx.x; // column index
    int y = blockDim.y * blockIdx.y + threadIdx.y; // row index

    if (x < N && y < M) {
        float sum = 0.0f;

        for (int i = 0; i < K; i++) {
            int index_A = y * K + i;
            int index_B = i * N + x;

            sum += A[index_A] * B[index_B];
        }
        C[y * N + x] = sum;
    }
}

void matmul_cpu(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int i = 0; i < K; i++) {
                sum += A[row * K + i] * B[i * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

// compare CPU and GPU results
void verify_results(const float* C_cpu, const float* C_gpu, int size) {
    float max_error = 0.0f;
    for (int i = 0; i < size; i++) {
        float error = fabs(C_cpu[i] - C_gpu[i]);
        if (error > max_error) {
            max_error = error;
        }
    }
    printf("Maximum output error against CPU: %f\n", max_error);
    if (max_error > 1e-3) {
        printf("WARNING: Results do not match!\n");
    }
    else {
        printf("SUCCESS: GPU results match CPU results.\n");
    }
}

int main() {
    srand(42); // standard seed
    dim3 threadsPerBlock(16, 16); // typical
    dim3 blocksPerGrid(
        (n + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (m + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    // calc size of matrices in bytes
    size_t bytesA = (size_t)m * k * sizeof(float);
    size_t bytesB = (size_t)k * n * sizeof(float);
    size_t bytesC = (size_t)m * n * sizeof(float);

    // allocate memory in ram
    float* A = (float*)malloc(bytesA); HOST_CHECK(A);
    float* B = (float*)malloc(bytesB); HOST_CHECK(B);
    float* C_gpu = (float*)malloc(bytesC); HOST_CHECK(C_gpu);
    float* C_cpu = (float*)malloc(bytesC); HOST_CHECK(C_cpu);

    // allocate mem in gpu
    float* d_A, * d_B, * d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));

    // fill matrices with random floats
    fill_random(A, m, k);
    fill_random(B, k, n);

    // copy matrices from ram to gpu
    CUDA_CHECK(cudaMemcpy(d_A, A, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, bytesB, cudaMemcpyHostToDevice));

    // warmup 
    matmul_kernel << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, m, k, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // benchmarking
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    // Launch kernel
    matmul_kernel << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, m, k, n);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop)); // Wait for GPU to finish

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // copy result from gpu to ram
    CUDA_CHECK(cudaMemcpy(C_gpu, d_C, bytesC, cudaMemcpyDeviceToHost));

   
    // Floating point operations: 2 * M * N * K
    double flops = 2.0 * m * n * k;
    double gflops = (flops / 1e9) / (milliseconds / 1000.0);

    printf("\n--- Performance Metrics ---\n");
    printf("Matrix Size: %dx%d * %dx%d\n", m, k, k, n);
    printf("Execution Time: %.3f ms\n", milliseconds);
    printf("Throughput: %.2f GFLOPS\n", gflops);
    printf("---------------------------\n\n");

    // --- VISUALIZATION & VALIDATION  (made w ai)
    visualize_matrix("Matrix A", A, m, k);
    visualize_matrix("Matrix B", B, k, n);
    visualize_matrix("Matrix C (GPU Result)", C_gpu, m, n);

    printf("Calculating CPU reference...\n");
    matmul_cpu(A, B, C_cpu, m, k, n);
    verify_results(C_cpu, C_gpu, m * n);

    // free all mem
    free(A);
    free(B);
    free(C_gpu);
    free(C_cpu);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    printf("\nProgram successfully completed.\n");
    return 0;
}