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
} while(0

#define HOST_CHECK(ptr) do { \
    if ((ptr) == nullptr) { \
        printf("host allocation failed at %s:%d\n", __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

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

// kernel func to add 2 vectors
__global__ void add_vectors(double* a, double* b, double* c) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id < Nc) {
		c[id] = a[id] + b[id];
	}
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

void matmul_cpu(const float* A, const float* B, float* C,
	int M, int K, int N) {
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

// vector addition from vecadd.cu
void run_vector_addition(int N) {
	size_t totalBytes = N * sizeof(double);

	// allocate mem in ram
	double* A = (double*)malloc(totalBytes);
	double* B = (double*)malloc(totalBytes);
	double* C = (double*)malloc(totalBytes);

	// allocate mem in gpu
	double* d_A, * d_B, * d_C;
	CUDA_CHECK(cudaMalloc(&d_A, totalBytes));
	CUDA_CHECK(cudaMalloc(&d_B, totalBytes));
	CUDA_CHECK(cudaMalloc(&d_C, totalBytes));


	// fill arrays A and B
	for (int i = 0; i < N; i++) {
		A[i] = i;
		B[i] = i * 2;
	}

	// copy data from ram to gpu via 
	CUDA_CHECK(cudaMemcpy(d_A, A, totalBytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_B, B, totalBytes, cudaMemcpyHostToDevice));

	int thr_per_blk = 256; // number of threads per block
	int blk_in_grid = (N + thr_per_blk - 1) / thr_per_blk; // number of blocks in grid

	// call kernel
	add_vectors << <blk_in_grid, thr_per_blk >> > (d_A, d_B, d_C);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	// copy result from gpu to ram
	CUDA_CHECK(cudaMemcpy(C, d_C, totalBytes, cudaMemcpyDeviceToHost));

	// check accuracy
	for (int i = 0; i < N; i++) {
		if (fabs(C[i] - (A[i] + B[i])) > 1e-9) {
			printf("\nError: value of C[%d] = %f instead of %f\n\n", i, C[i], A[i] + B[i]);
			exit(1);
		}
	}

	printf("Threads per block: %d\n", thr_per_blk);
	printf("Blocks in grid: %d\n", blk_in_grid);
	printf("Vector addition completed successfully.\n");

	// free all mem
	free(A);
	free(B);
	free(C);
	CUDA_CHECK(cudaFree(d_A));
	CUDA_CHECK(cudaFree(d_B));
	CUDA_CHECK(cudaFree(d_C));
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
	float* C = (float*)malloc(bytesC); HOST_CHECK(C);
	
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

	// perform matrix multiplication on gpu
	matmul_kernel << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, m, k, n);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	// copy result from gpu to ram
	CUDA_CHECK(cudaMemcpy(C, d_C, bytesC, cudaMemcpyDeviceToHost));

	// print result matrix C
	//for (int i = 0; i < m; i++) {
	//	for (int j = 0; j < n; j++) {
	//		printf("%f ", C[i * n + j]);
	//	}
	//	printf("\n");
	//}

	// free all mem
	free(A);
	free(B);
	free(C);
	CUDA_CHECK(cudaFree(d_A));
	CUDA_CHECK(cudaFree(d_B));
	CUDA_CHECK(cudaFree(d_C));

	printf("Program successfully completed.\n");
}