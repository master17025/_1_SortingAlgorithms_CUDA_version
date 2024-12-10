#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

__global__ void InitCurandStates(curandState* states, long long seed, long int NumberOfElements);
__global__ void GenerateRandomArrayKernel(int* d_array, curandState* states, int lowerBound, int upperBound, long int NumberOfElements);
int* CreateRandomArray(long int NumberOfElements, int lowerBound, int upperBound);
