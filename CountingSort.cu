#include "CountingSort.cuh"
#include <iostream>
#include <chrono>  // For measuring execution time

#include <cub/cub.cuh>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
void countingSort(int upperBound, long int NumberOfElements, int* inputArray)
{
    // Step 1: The inputArray has `NumberOfElements` elements to sort,
    // and `upperBound` specifies the maximum value in the array.

    // Step 2: Create an output array to store the sorted elements.
    // Dynamically allocate memory for the output array.
    int* outputArray = new int[NumberOfElements]();

    // Step 3: Define the range of possible values.
    int range = upperBound + 1;

    // Step 4: Create a count array to store the frequency of each element.
    // Dynamically allocate memory for the count array and initialize to 0.
    int* countArray = new int[range]();

    // Step 5: Count the occurrences of each element in the inputArray.
    for (long int i = 0; i < NumberOfElements; ++i)
        ++countArray[inputArray[i]];

    // Step 6: Modify the count array to store cumulative counts.
    for (int i = 1; i < range; ++i)
        countArray[i] += countArray[i - 1];

    // Step 7: Place elements from the input array into the output array using the count array.
    for (long int i = NumberOfElements - 1; i >= 0; --i) {
        int index = --countArray[inputArray[i]];
        outputArray[index] = inputArray[i];
    }

    // Step 8: Transfer the sorted values from the output array back to the input array.
    for (long int i = 0; i < NumberOfElements; ++i)
        inputArray[i] = outputArray[i];

    // Step 9: Free the dynamically allocated memory.
    delete[] outputArray;
    delete[] countArray;
}


#define threadsperblock 512

// Kernel to initialize an array
__global__ void InitializeArray(int* array, int size, int value) {
    long int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        array[tid] = value;
    }
}

// Kernel to count occurrences of each element
__global__ void CountOccurrences(int* inputArray, int* countArray, long int NumberOfElements) {
    long int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NumberOfElements) {
        atomicAdd(&countArray[inputArray[tid]], 1);
    }
}

// Kernel to place elements in the correct position
__global__ void PlaceElements(int* inputArray, int* outputArray, int* countArray, long int NumberOfElements) {
    long int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NumberOfElements) {
        int value = inputArray[tid];
        int position = atomicSub(&countArray[value], 1) - 1;
        outputArray[position] = value;
    }
}


// Kernel to place elements in the correct position
__global__ void PlaceElements(int* inputArray, int* outputArray, int* countArray, int NumberOfElements) {
    long int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NumberOfElements) {
        int value = inputArray[tid];
        int position = atomicSub(&countArray[value], 1) - 1;
        outputArray[position] = value;
    }
}

// Counting Sort function using CUDA
void CountingSortGPU(int upperBound, long int NumberOfElements, int* inputArray) {
    int* d_inputArray;
    int* d_outputArray;
    int* d_countArray;

    int range = upperBound + 1;

    // Allocate device memory
    cudaMalloc(&d_inputArray, sizeof(int) * NumberOfElements);
    cudaMalloc(&d_outputArray, sizeof(int) * NumberOfElements);
    cudaMalloc(&d_countArray, sizeof(int) * range);




    // Copy input data to device
    cudaMemcpy(d_inputArray, inputArray, sizeof(int) * NumberOfElements, cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();

    // Initialize the count array on the device
    int blocksPerGridCount = (range + threadsperblock - 1) / threadsperblock;
    InitializeArray << <blocksPerGridCount, threadsperblock >> > (d_countArray, range, 0);

    // Count occurrences of each element
    long int blocksPerGridInput = (NumberOfElements + threadsperblock - 1) / threadsperblock;
    CountOccurrences << <blocksPerGridInput, threadsperblock >> > (d_inputArray, d_countArray, NumberOfElements);

    // Compute cumulative counts using CUB InclusiveSum
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Get temporary storage size for CUB InclusiveSum
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_countArray, d_countArray, range);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Perform cumulative sum
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_countArray, d_countArray, range);

    // Place elements in the correct position
    PlaceElements << <blocksPerGridInput, threadsperblock >> > (d_inputArray, d_outputArray, d_countArray, NumberOfElements);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Copy sorted array back to host
    cudaMemcpy(inputArray, d_outputArray, sizeof(int) * NumberOfElements, cudaMemcpyDeviceToHost);




    std::cout << "Time taken to sort using Counting Sort on GPU: " << duration.count() << " ms" << std::endl;

    // Free device memory
    cudaFree(d_temp_storage);
    cudaFree(d_inputArray);
    cudaFree(d_outputArray);
    cudaFree(d_countArray);
}

