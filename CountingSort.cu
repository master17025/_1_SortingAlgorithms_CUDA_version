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

// Kernel to initialize an array
__global__ void InitializeVector(int* vector, int size, int value) {
    long int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        vector[tid] = value;
    }
}

// Kernel to count occurrences of digits (single-pass for Counting Sort)
__global__ void CountDigitsKernel(const int* input, int* count, int size, int range) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        int digit = input[tid] % range; // Only single-digit range
        atomicAdd(&count[digit], 1);
    }
}

// Kernel to place elements in sorted positions
__global__ void PlaceElementsKernel(const int* input, int* output, int* count, int size, int range) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        int digit = input[tid] % range; // Only single-digit range
        int position = atomicSub(&count[digit], 1) - 1;
        output[position] = input[tid];
    }
}
// Single-pass Counting Sort function on GPU
void CountingSortGPU(std::vector<int>& h_input, int upperBound) {
    const int range = upperBound + 1; // Range of values [0, upperBound]
    int size = h_input.size();

    // GPU memory pointers
    int* d_input, * d_output;
    cudaMalloc(&d_input, size * sizeof(int));
    cudaMalloc(&d_output, size * sizeof(int));

    // Thrust device vector for count array
    thrust::device_vector<int> d_count(range, 0);

    

    // Copy the input vector to the GPU
    cudaMemcpy(d_input, h_input.data(), size * sizeof(int), cudaMemcpyHostToDevice);


    auto start = std::chrono::high_resolution_clock::now();
    // Step 1: Count occurrences of each element
    int blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    CountDigitsKernel << <blocks, THREADS_PER_BLOCK >> > (d_input, thrust::raw_pointer_cast(d_count.data()), size, range);
    cudaDeviceSynchronize();

    // Step 2: Compute cumulative sums using Thrust inclusive scan
    thrust::inclusive_scan(d_count.begin(), d_count.end(), d_count.begin());

    // Step 3: Place elements into the output array
    PlaceElementsKernel << <blocks, THREADS_PER_BLOCK >> > (d_input, d_output, thrust::raw_pointer_cast(d_count.data()), size, range);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();


    // Copy sorted data back to the host
    cudaMemcpy(h_input.data(), d_output, size * sizeof(int), cudaMemcpyDeviceToHost);

    
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "GPU Counting Sort Time: " << duration.count() << " ms" << std::endl;

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);
}