#include "RadixSort.cuh"

#include <iostream>
#include <chrono>  // For measuring execution time

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// Counting sort for radix sort: sorts based on individual digit
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

void countingSortRadix(int divider, long int NumberOfElements, std::vector<int>& inputVector)
{
    // Step 1: Create an output vector to store sorted elements
    std::vector<int> outputVector(NumberOfElements, 0);

    // Step 2: Define the range of digits (0-9) for the decimal system
    int range = 10;

    // Step 3: Create a count vector to store the frequency of each digit (0-9)
    std::vector<int> countVector(range, 0);

    // Step 4: Count the occurrences of each digit at the current digit place (based on divider)
    for (long int i = 0; i < NumberOfElements; i++) {
        int digit = (inputVector[i] / divider) % range;
        countVector[digit]++;
    }

    // Step 5: Modify the count vector to store the cumulative count
    for (int i = 1; i < range; i++) {
        countVector[i] += countVector[i - 1];
    }

    // Step 6: Build the output vector by placing the elements in sorted order
    for (long int i = NumberOfElements - 1; i >= 0; i--) {
        int digit = (inputVector[i] / divider) % range;
        outputVector[countVector[digit] - 1] = inputVector[i];
        countVector[digit]--;
    }

    // Step 7: Copy the sorted values back into the original input vector
    for (long int i = 0; i < NumberOfElements; i++) {
        inputVector[i] = outputVector[i];
    }
}


// Function prototype for countingSortRadix

// Main radix sort function: sorts the entire vector
void RadixSort(long int NumberOfElements, std::vector<int>& inputVector)
{
    // Find the maximum element in the input vector to determine the number of digits
    int maximum = *std::max_element(inputVector.begin(), inputVector.end());

    // Step 1: Call countingSortRadix for each digit place (1s, 10s, 100s, etc.)
    for (int divider = 1; maximum / divider > 0; divider *= 10) {
        countingSortRadix(divider, NumberOfElements, inputVector);
    }
}

// CUDA kernel to calculate digit frequencies
__global__ void countingKernel(int* input, int* count, int n, int divider) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int digit = (input[idx] / divider) % 10;
        atomicAdd(&count[digit], 1);  // Atomic addition to ensure thread safety
    }
}

// CUDA kernel to place elements into the output array
__global__ void outputKernel(int* input, int* output, int* count, int n, int divider) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int digit = (input[idx] / divider) % 10;
        int pos = atomicAdd(&count[digit], 1);
        output[pos] = input[idx];
    }
}

#define THREADS_PER_BLOCK 1024

// Host function for Radix Sort
std::chrono::duration<double, std::milli> RadixSortGPU(std::vector<int>& inputVector) {
    int n = inputVector.size();
    int* d_input = nullptr, * d_output = nullptr, * d_count = nullptr;

    // Step 1: Allocate memory on the GPU
    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));
    cudaMalloc(&d_count, 10 * sizeof(int));

    // Copy input data to GPU
    cudaMemcpy(d_input, inputVector.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    thrust::device_ptr<int> thrust_count(d_count);

    auto start = std::chrono::high_resolution_clock::now();

    // Find the maximum value in the input array to determine the number of digits
    int maxValue = *std::max_element(inputVector.begin(), inputVector.end());

    // Perform sorting for each digit (1's, 10's, 100's, etc.)
    for (int divider = 1; maxValue / divider > 0; divider *= 10) {
        // Step 2: Reset count array to zero
        cudaMemset(d_count, 0, 10 * sizeof(int));

        // Step 3: Launch counting kernel
        int gridSize = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        countingKernel << <gridSize, THREADS_PER_BLOCK >> > (d_input, d_count, n, divider);
        cudaDeviceSynchronize();

        // Step 4: Compute prefix sums using Thrust

        thrust::exclusive_scan(thrust_count, thrust_count + 10, thrust_count);
        cudaDeviceSynchronize();

        // Step 5: Launch output kernel to place elements in the output array
        cudaMemset(d_output, 0, n * sizeof(int));
        outputKernel << <gridSize, THREADS_PER_BLOCK >> > (d_input, d_output, d_count, n, divider);
        cudaDeviceSynchronize();

        // Step 6: Copy output back to input for the next iteration
        cudaMemcpy(d_input, d_output, n * sizeof(int), cudaMemcpyDeviceToDevice);
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;

    // Copy the sorted result back to the CPU
    cudaMemcpy(inputVector.data(), d_output, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_count);
    return duration;
}
