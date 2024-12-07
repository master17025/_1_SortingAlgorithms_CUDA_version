#include "CountingSort.cuh"
#include <iostream>
#include <chrono>  // For measuring execution time

#include <cub/cub.cuh>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
void countingSort(int upperBound, int NumberOfElements, int* inputArray)
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
    for (int i = 0; i < NumberOfElements; ++i)
        ++countArray[inputArray[i]];

    // Step 6: Modify the count array to store cumulative counts.
    for (int i = 1; i < range; ++i)
        countArray[i] += countArray[i - 1];

    // Step 7: Place elements from the input array into the output array using the count array.
    for (int i = NumberOfElements - 1; i >= 0; --i) {
        outputArray[--countArray[inputArray[i]]] = inputArray[i];
    }

    // Step 8: Transfer the sorted values from the output array back to the input array.
    for (int i = 0; i < NumberOfElements; ++i)
        inputArray[i] = outputArray[i];

    // Step 9: Free the dynamically allocated memory.
    delete[] outputArray;
    delete[] countArray;
}

__global__ void CumulativeSum(int* inputVector, int NumberOfElements) {
    extern __shared__ int sharedData[];

    int tx = threadIdx.x;
    int tid = blockIdx.x * blockDim.x + tx;

    if (tid >= NumberOfElements) return;

    // Load data into shared memory
    sharedData[tx] = inputVector[tid];
    __syncthreads();

    // Perform cumulative sum in shared memory
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        int temp = 0;
        if (tx >= offset) {
            temp = sharedData[tx - offset];
        }
        __syncthreads();
        sharedData[tx] += temp;
        __syncthreads();
    }

    // Write results back to global memory
    inputVector[tid] = sharedData[tx];
}

void CumulativeSumCUB(int* randomList, int NumberOfElements) {
    // Allocate device memory
    int* d_input;
    int* d_output;
    cudaMalloc(&d_input, sizeof(int) * NumberOfElements);
    cudaMalloc(&d_output, sizeof(int) * NumberOfElements);


    // Start the timer to measure execution time
    auto start = std::chrono::high_resolution_clock::now();
    // Copy input data to device
    cudaMemcpy(d_input, randomList, sizeof(int) * NumberOfElements, cudaMemcpyHostToDevice);

    // Temporary storage allocation
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Get temporary storage size
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_input, d_output, NumberOfElements);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Perform cumulative sum
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_input, d_output, NumberOfElements);

    // Stop the timer after sorting is complete
    auto end = std::chrono::high_resolution_clock::now();


    // Copy result back to host
    cudaMemcpy(randomList, d_output, sizeof(int) * NumberOfElements, cudaMemcpyDeviceToHost);


    //printArray(randomList, NumberOfElements);
    // Calculate the time duration in milliseconds
    std::chrono::duration<double, std::milli> duration = end - start;

    // Output the time taken to sort using Radix Sort
    std::cout << "Time taken for cumulative sum in cublas: " << duration.count() << " milliseconds" << std::endl;

    // Free memory
    cudaFree(d_temp_storage);
    cudaFree(d_input);
    cudaFree(d_output);
}



