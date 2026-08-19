#include <cstdio>
#include <cuda_runtime.h>


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
	cudaMalloc(&d_A, totalBytes);
	cudaMalloc(&d_B, totalBytes);
	cudaMalloc(&d_C, totalBytes);


	// fill arrays A and B
	for (int i = 0; i < N; i++) {
		A[i] = i;
		B[i] = i * 2;
	}

	// copy data from ram to gpu via pcie
	cudaMemcpy(d_A, A, totalBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, totalBytes, cudaMemcpyHostToDevice);

	int thr_per_blk = 256; // number of threads per block
	int blk_in_grid = (N + thr_per_blk - 1) / thr_per_blk; // number of blocks in grid

	// call kernel
	add_vectors << <blk_in_grid, thr_per_blk >> > (d_A, d_B, d_C);

	// copy result from gpu to ram
	cudaMemcpy(C, d_C, totalBytes, cudaMemcpyDeviceToHost);

	// check accuracy
	for (int i = 0; i < N; i++) {
		if (C[i] != A[i] + B[i]) {
			printf("Error at index %d: %f != %f + %f\n", i, C[i], A[i], B[i]);
			break;
		}
	}

	// free all mem
	free(A);
	free(B);
	free(C);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	printf("Threads per block: %d\n", thr_per_blk);
	printf("Blocks in grid: %d\n", blk_in_grid);
	printf("Vector addition completed successfully.\n");
}