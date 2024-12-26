#include "CountingSort.cuh"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/scatter.h>
#include <thrust/execution_policy.h>
#include <vector>
#include <chrono>
#include <iostream>

void countingSort(int upperBound, long int NumberOfElements, std::vector<int>& inputArray)
{
    // Step 1: Define the range of possible values.
    int range = upperBound + 1;

    // Step 2: Create a count vector to store the frequency of each element.
    std::vector<int> countArray(range, 0);

    // Step 3: Count the occurrences of each element in the input array.
    for (long int i = 0; i < NumberOfElements; ++i) {
        ++countArray[inputArray[i]];
    }

    // Step 4: Modify the count vector to store cumulative counts.
    for (int i = 1; i < range; ++i) {
        countArray[i] += countArray[i - 1];
    }

    // Step 5: Create an output vector to store the sorted elements.
    std::vector<int> outputArray(NumberOfElements);

    // Step 6: Place elements from the input array into the output vector using the count vector.
    for (long int i = NumberOfElements - 1; i >= 0; --i) {
        int index = --countArray[inputArray[i]];
        outputArray[index] = inputArray[i];
    }

    // Step 7: Copy the sorted values back to the input vector.
    inputArray = outputArray;
}

#define THREADS_PER_BLOCK  1024

__global__ void countKernel(const int* inputVector, int* countArray, long int NumberOfElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NumberOfElements) {
        atomicAdd(&countArray[inputVector[idx]], 1);
    }
}


__global__ void placeKernel(const int* inputVector, int* countArray, int* outputArray, long int NumberOfElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NumberOfElements) {
        int value = inputVector[idx];
        int pos = atomicSub(&countArray[value], 1) - 1;
        outputArray[pos] = value;
    }
}
// Single-pass Counting Sort function on GPU
// Single-pass Counting Sort function on GPU
std::chrono::duration<double, std::milli> CountingSortGPU(std::vector<int>& h_input, int upperBound) {
    const int range = upperBound + 1; // Range of values [0, upperBound]
    int size = h_input.size();

    // GPU memory pointers
    int* d_input, * d_output, * d_count;
    cudaMalloc(&d_input, size * sizeof(int));
    cudaMalloc(&d_output, size * sizeof(int));
    cudaMalloc(&d_count, range * sizeof(int));
    thrust::device_ptr<int> thrust_countArray(d_count);

    // Initialize count array to zero
    cudaMemset(d_count, 0, range * sizeof(int));

    

    // Copy the input vector to the GPU
    cudaMemcpy(d_input, h_input.data(), size * sizeof(int), cudaMemcpyHostToDevice);

    
    auto start = std::chrono::high_resolution_clock::now();
    // Step 1: Count occurrences of each element
    int blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    countKernel << <blocks, THREADS_PER_BLOCK >> > (d_input, d_count, size);
    cudaDeviceSynchronize();

    // Step 2: Compute cumulative sums using custom kernel
        // Perform prefix sum using Thrust
    
    thrust::inclusive_scan(thrust_countArray, thrust_countArray + range, thrust_countArray);
    cudaDeviceSynchronize();

    // Step 3: Place elements into the output array
    placeKernel << <blocks, THREADS_PER_BLOCK >> > (d_input, d_count, d_output, size);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    // Copy sorted data back to the host
    cudaMemcpy(h_input.data(), d_output, size * sizeof(int), cudaMemcpyDeviceToHost);
    

    std::chrono::duration<double, std::milli> duration = end - start;

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_count);

    return duration;
}

