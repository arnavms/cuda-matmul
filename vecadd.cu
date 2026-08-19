#include <cstdio>
#include <cuda_runtime.h>
#include <cmath>

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

#define N 1000000

// kernel func to add 2 vectors

__global__ void add_vectors(double* a, double* b, double* c) {
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id < N) {
		c[id] = a[id] + b[id];
	}
}

int main() {
	size_t totalBytes = N * sizeof(double);

	// allocate mem in ram
	double* A = (double*)malloc(totalBytes);
	double* B = (double*)malloc(totalBytes);
	double* C = (double*)malloc(totalBytes);

	// allocate mem in gpu
	double *d_A, *d_B, *d_C;
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