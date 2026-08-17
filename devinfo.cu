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
	
	return 0;
}