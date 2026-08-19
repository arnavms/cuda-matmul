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


void __global__ threadIdentity() {
	printf("Thread ID: %d\n", threadIdx.x);
}

int main() {
	int count;
	cudaError_t err = cudaGetDeviceCount(&count);
	if (err != cudaSuccess) {
		printf("Error getting device count: %s\n", cudaGetErrorString(err));
		return 1;
	}

	printf("CUDA devices found: %d\n", count);
	for (int i = 0; i < count; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device name: %s\n", prop.name);
		printf("  Compute capability   : %d.%d\n", prop.major, prop.minor);
		printf("  SM count             : %d\n", prop.multiProcessorCount);
		printf("  Global memory        : %.2f GB\n", prop.totalGlobalMem / 1073741824.0);
		printf("  Shared mem per block : %.1f KB\n", prop.sharedMemPerBlock / 1024.0);
		printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("  Warp size            : %d\n", prop.warpSize);


		int clockKHz = 0, buswidth = 0;
		cudaDeviceGetAttribute(&clockKHz, cudaDevAttrClockRate, i);
		cudaDeviceGetAttribute(&buswidth, cudaDevAttrGlobalMemoryBusWidth, i);
		printf("  Clock rate           : %d kHz\n", clockKHz);
		printf("  Bus width            : %d bits\n", buswidth);
	}
	

	threadIdentity << <1, 4 >> > ();

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error launching kernel: %s\n", cudaGetErrorString(err));
		return 1;
	}

	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		printf("Error synchronizing device: %s\n", cudaGetErrorString(err));
		return 1;
	}
	
	printf("Thread Kernel executed successfully.\n");

	return 0;
}