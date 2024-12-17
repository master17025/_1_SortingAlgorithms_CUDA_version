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

#define THREADS_PER_BLOCK 1024

// CUDA kernel to calculate digit frequencies
__global__ void countingKernel(int* input, int* count, int n, int divider) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int digit = (input[idx] / divider) % 10;
        atomicAdd(&count[digit], 1);  // Atomic addition to ensure thread safety
    }
}

// CUDA kernel to place elements into the output array
__global__ void outputKernel(int* input, int* output, int* count, int* prefixSum, int n, int divider) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int digit = (input[idx] / divider) % 10;
        int pos = atomicAdd(&prefixSum[digit], 1);
        output[pos] = input[idx];
    }
}



// Host function for radix sort
void RadixSortGPU(std::vector<int>& inputVector) {
    int n = inputVector.size();
    int* d_input, * d_output, * d_count;

    // Step 1: Allocate memory on the GPU
    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));
    cudaMalloc(&d_count, 10 * sizeof(int));

    // Copy input data to GPU
    cudaMemcpy(d_input, inputVector.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();

    int maxValue = *std::max_element(inputVector.begin(), inputVector.end());

    for (int divider = 1; maxValue / divider > 0; divider *= 10) {
        // Step 2: Reset count array to 0
        cudaMemset(d_count, 0, 10 * sizeof(int));

        // Step 3: Launch counting kernel
        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;
        countingKernel << <gridSize, blockSize >> > (d_input, d_count, n, divider);
        cudaDeviceSynchronize();

        // Step 4: Use Thrust for prefix sum (exclusive scan)
        thrust::device_vector<int> d_countVector(10);
        thrust::device_vector<int> d_prefixSum(10);

        // Copy count data to thrust device vector
        cudaMemcpy(thrust::raw_pointer_cast(d_countVector.data()), d_count, 10 * sizeof(int), cudaMemcpyDeviceToDevice);

        // Compute prefix sums using exclusive scan
        thrust::exclusive_scan(thrust::device, d_countVector.begin(), d_countVector.end(), d_prefixSum.begin());

        // Step 5: Place elements into output array
        cudaMemset(d_output, 0, n * sizeof(int));
        outputKernel << <gridSize, blockSize >> > (d_input, d_output, thrust::raw_pointer_cast(d_countVector.data()),
            thrust::raw_pointer_cast(d_prefixSum.data()), n, divider);
        cudaDeviceSynchronize();

        // Step 6: Copy output back to input for next iteration
        cudaMemcpy(d_input, d_output, n * sizeof(int), cudaMemcpyDeviceToDevice);
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "GPU Radix Sort Time: " << duration.count() << " ms" << std::endl;

    // Copy the sorted result back to the CPU
    cudaMemcpy(inputVector.data(), d_output, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_count);
}