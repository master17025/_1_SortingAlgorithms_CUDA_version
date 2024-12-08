#include"RandomGeneration.cuh"

// Kernel to initialize CURAND states
__global__ void InitCurandStates(curandState* states, unsigned long seed, int NumberOfElements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NumberOfElements) {
        curand_init(seed, tid, 0, &states[tid]);
    }
}

// Kernel to generate random numbers using pre-initialized states
__global__ void GenerateRandomArrayKernel(int* d_array, curandState* states, int lowerBound, int upperBound, int NumberOfElements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NumberOfElements) {
        curandState localState = states[tid];  // Use pre-initialized state
        float randomValue = curand_uniform(&localState); // Generate random float in range (0, 1]
        d_array[tid] = lowerBound + (int)((upperBound - lowerBound + 1) * randomValue);
        states[tid] = localState; // Save updated state
    }
}

// Function to generate a random array using CUDA
int* CreateRandomArray(int NumberOfElements, int lowerBound, int upperBound) {
    int* d_array;
    int* h_array = new int[NumberOfElements];
    curandState* d_states;

    // Allocate device memory
    cudaMalloc(&d_array, sizeof(int) * NumberOfElements);
    cudaMalloc(&d_states, sizeof(curandState) * NumberOfElements);

    // Configure kernel
    int blocksPerGrid = (NumberOfElements + threadsperblock - 1) / threadsperblock;
    unsigned long seed = time(0);

    // Initialize CURAND states
    InitCurandStates << <blocksPerGrid, threadsperblock >> > (d_states, seed, NumberOfElements);

    // Generate random numbers using pre-initialized states
    GenerateRandomArrayKernel << <blocksPerGrid, threadsperblock >> > (d_array, d_states, lowerBound, upperBound, NumberOfElements);

    // Copy the generated random numbers back to host memory
    cudaMemcpy(h_array, d_array, sizeof(int) * NumberOfElements, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_array);
    cudaFree(d_states);

    return h_array;
}