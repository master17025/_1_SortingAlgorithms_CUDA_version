#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "VectorFunc.h"    // Include custom array functions (e.g., CreateRandomArray)
#include "CountingSort.cuh"  // Include Counting Sort implementation
#include "RadixSort.h"     // Include Radix Sort implementation

#include <iostream>
#include <chrono>  // For measuring execution time

#include <cub/cub.cuh>
#include <curand_kernel.h>

// Define the number of elements of the integer array
int const NumberOfElements = 1e7; // Example: 450 million elements
#define threadsperblock 1024

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

// Function to measure the time and performance of Counting Sort
void CountingSortAnalysis(int* randomList, int lowerBound, int upperBound) {
    auto start = std::chrono::high_resolution_clock::now();
    countingSort(upperBound, NumberOfElements, randomList);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Time taken to sort the list using Counting sort: " << duration.count() << " milliseconds" << std::endl;
}

// Function to measure the time and performance of Radix Sort
void RadixSortAnalysis(int* randomList, int lowerBound, int upperBound) {
    auto start = std::chrono::high_resolution_clock::now();
    RadixSort(NumberOfElements, randomList);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Time taken to sort the list using Radix sort: " << duration.count() << " milliseconds" << std::endl;
}

// Function to measure the time and performance of Radix Sort
void CumulativeSumAnalysis(int* randomList, int lowerBound, int upperBound)
{
    size_t bytes = sizeof(int) * NumberOfElements;

    // Device vector pointers
    int* d_randomList;

    // Allocate device memory (GPU)
    cudaMalloc(&d_randomList, bytes);
    int blocksPerGrid = (NumberOfElements + (threadsperblock - 1)) / threadsperblock;



    // Copy to device
    cudaMemcpy(d_randomList, randomList, bytes, cudaMemcpyHostToDevice);
    //printArray(randomList, NumberOfElements);


        // Start the timer to measure execution time
    auto start = std::chrono::high_resolution_clock::now();

    // Call kernel
    CumulativeSum << <blocksPerGrid, threadsperblock >> > (d_randomList, NumberOfElements);

    // Stop the timer after sorting is complete
    auto end = std::chrono::high_resolution_clock::now();

    // copy data from device memory to host memory (CPU to  GPU)
    cudaMemcpy(randomList, d_randomList, bytes, cudaMemcpyDeviceToHost);



    //printArray(randomList, NumberOfElements);
    // Calculate the time duration in milliseconds
    std::chrono::duration<double, std::milli> duration = end - start;

    // Output the time taken to sort using Radix Sort
    std::cout << "Time taken for cumulative sum: " << duration.count() << " milliseconds" << std::endl;
}



int main() {
    int lowerBound = 1;
    int upperBound = 9;

    // Generate random array on GPU
    int* h_randomList = CreateRandomArray(NumberOfElements, lowerBound, upperBound);

    // Perform analysis
    //CumulativeSumAnalysis(h_randomList, lowerBound, upperBound);
    //CountingSortAnalysis(h_randomList, lowerBound, upperBound);
    //RadixSortAnalysis(h_randomList, lowerBound, upperBound);

    // Free host memory
    delete[] h_randomList;

    return 0;
}
