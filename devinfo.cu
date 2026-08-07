#include <cstdio>
#include <cuda_runtime.h>

int main() {
	int count;
	cudaError_t err = cudaGetDeviceCount(&count);
	if (err != cudaSuccess) {
		printf("Error getting device count: %s\n", cudaGetErrorString(err));
		return 1;
	}

	printf("CUDA devices found: %d\n", count);
	return 0;
}