#include "CountingSort.cuh"
#include <iostream>
#include <chrono>  // For measuring execution time
#include <vector>
#include <cub/cub.cuh>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
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

#define threadsperblock 1024

// Kernel to initialize an array
__global__ void InitializeVector(int* vector, int size, int value) {
    long int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        vector[tid] = value;
    }
}

// Kernel to count occurrences of each element
__global__ void CountOccurrences(const int* inputVector, int* countVector, long int NumberOfElements) {
    long int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NumberOfElements) {
        atomicAdd(&countVector[inputVector[tid]], 1);
    }
}


// Kernel to place elements in the correct position
__global__ void PlaceElements(const int* inputVector, int* outputVector, int* countVector, long int NumberOfElements) {
    long int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NumberOfElements) {
        int value = inputVector[tid];
        int position = atomicSub(&countVector[value], 1) - 1;
        outputVector[position] = value;
    }
}


// Counting Sort function using CUDA
void CountingSortGPU(int upperBound, const std::vector<int>& inputVector, std::vector<int>& outputVector) {
    long int NumberOfElements = inputVector.size();
    int range = upperBound + 1;

    // Device pointers
    int* d_inputVector;
    int* d_outputVector;
    int* d_countVector;

    // Allocate device memory
    cudaMalloc(&d_inputVector, sizeof(int) * NumberOfElements);
    cudaMalloc(&d_outputVector, sizeof(int) * NumberOfElements);
    cudaMalloc(&d_countVector, sizeof(int) * range);

    // Copy input data to the device
    cudaMemcpy(d_inputVector, inputVector.data(), sizeof(int) * NumberOfElements, cudaMemcpyHostToDevice);

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Step 1: Initialize count vector
    int blocksPerGridCount = (range + threadsperblock - 1) / threadsperblock;
    InitializeVector << <blocksPerGridCount, threadsperblock >> > (d_countVector, range, 0);

    // Step 2: Count occurrences
    long int blocksPerGridInput = (NumberOfElements + threadsperblock - 1) / threadsperblock;
    CountOccurrences << <blocksPerGridInput, threadsperblock >> > (d_inputVector, d_countVector, NumberOfElements);

    // Step 3: Compute cumulative counts using CUB InclusiveSum
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Get temporary storage size and allocate it
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_countVector, d_countVector, range);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Perform cumulative sum
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_countVector, d_countVector, range);

    // Step 4: Place elements into the output vector
    PlaceElements << <blocksPerGridInput, threadsperblock >> > (d_inputVector, d_outputVector, d_countVector, NumberOfElements);

    // Step 5: Copy the sorted data back to the host
    cudaMemcpy(outputVector.data(), d_outputVector, sizeof(int) * NumberOfElements, cudaMemcpyDeviceToHost);

    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "Time taken to sort using Counting Sort on GPU: " << duration.count() << " ms" << std::endl;

    // Free device memory
    cudaFree(d_temp_storage);
    cudaFree(d_inputVector);
    cudaFree(d_outputVector);
    cudaFree(d_countVector);
}

