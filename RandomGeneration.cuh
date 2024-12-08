#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#define threadsperblock 1024
__global__ void InitCurandStates(curandState* states, unsigned long seed, int NumberOfElements);
__global__ void GenerateRandomArrayKernel(int* d_array, curandState* states, int lowerBound, int upperBound, int NumberOfElements);
int* CreateRandomArray(int NumberOfElements, int lowerBound, int upperBound);
