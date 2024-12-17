#include "RadixSort.cuh"

#include <iostream>
#include <chrono>  // For measuring execution time

#include <cub/cub.cuh>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// Counting sort for radix sort: sorts based on individual digit
#include <vector>

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

#define threadsperblock 1024

// Kernel to count the occurrences of digits
__global__ void CountDigitsKernel(const int* input, int* count, int size, int divider, int range) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        int digit = (input[tid] / divider) % range;
        atomicAdd(&count[digit], 1); // Atomic increment to avoid race conditions
    }
}

// Kernel to place sorted elements in the output array
__global__ void PlaceElementsKernel(const int* input, int* output, int* count, int size, int divider, int range) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        int digit = (input[tid] / divider) % range;
        int position = atomicSub(&count[digit], 1) - 1;
        output[position] = input[tid];
    }
}

// Main Radix Sort function on GPU using std::vector<int>
void RadixSortGPU(std::vector<int>& h_input) {
    const int range = 10; // Decimal range (0-9)
    int size = h_input.size();

    // GPU memory pointers
    int* d_input, * d_output, * d_count;

    // Allocate memory on the GPU
    cudaMalloc(&d_input, size * sizeof(int));
    cudaMalloc(&d_output, size * sizeof(int));
    cudaMalloc(&d_count, range * sizeof(int));

    // Find the maximum element to determine the number of digits
    int maxElement = *std::max_element(h_input.begin(), h_input.end());

    // Copy the input vector to the device
    cudaMemcpy(d_input, h_input.data(), size * sizeof(int), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();

    // Loop over each digit place (1s, 10s, 100s, etc.)
    for (int divider = 1; maxElement / divider > 0; divider *= 10) {

        // Initialize count array to zero
        cudaMemset(d_count, 0, range * sizeof(int));

        // Count digit occurrences in parallel
        int blocks = (size + threadsperblock - 1) / threadsperblock;
        CountDigitsKernel << <blocks, threadsperblock >> > (d_input, d_count, size, divider, range);
        cudaDeviceSynchronize();

        // Compute cumulative sums on the host
        std::vector<int> h_count(range, 0);
        cudaMemcpy(h_count.data(), d_count, range * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 1; i < range; i++) {
            h_count[i] += h_count[i - 1];
        }
        cudaMemcpy(d_count, h_count.data(), range * sizeof(int), cudaMemcpyHostToDevice);

        // Place elements into the output array in parallel
        PlaceElementsKernel << <blocks, threadsperblock >> > (d_input, d_output, d_count, size, divider, range);
        cudaDeviceSynchronize();

        // Swap pointers to prepare for the next iteration
        std::swap(d_input, d_output);
    }
}