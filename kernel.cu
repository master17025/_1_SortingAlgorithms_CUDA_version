#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "VectorFunc.h"    // Include custom array functions (e.g., CreateRandomArray)
#include "CountingSort.h"  // Include Counting Sort implementation
#include "RadixSort.h"     // Include Radix Sort implementation

#include <iostream>
#include <chrono>  // For measuring execution time
// Define the number of elements of the integer array
int const NumberOfElements = 8; // Example: 450 million elements

// Function to measure the time and performance of Counting Sort
void CountingSortAnalysis(int* randomList, int lowerBound, int upperBound)
{
    // Start the timer to measure execution time
    auto start = std::chrono::high_resolution_clock::now();

    // Perform Counting Sort on the random list
    countingSort(upperBound, NumberOfElements, randomList);

    // Stop the timer after sorting is complete
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the time duration in milliseconds
    std::chrono::duration<double, std::milli> duration = end - start;

    // Output the time taken to sort using Counting Sort
    std::cout << "Time taken to sort the list using Counting sort: " << duration.count() << " milliseconds" << std::endl;
}

// Function to measure the time and performance of Radix Sort
void RadixSortAnalysis(int* randomList, int lowerBound, int upperBound)
{
    // Start the timer to measure execution time
    auto start = std::chrono::high_resolution_clock::now();

    // Perform Radix Sort on the random list
    RadixSort(NumberOfElements, randomList);

    // Stop the timer after sorting is complete
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the time duration in milliseconds
    std::chrono::duration<double, std::milli> duration = end - start;

    // Output the time taken to sort using Radix Sort
    std::cout << "Time taken to sort the list using Radix sort: " << duration.count() << " milliseconds" << std::endl;
}

#define threadsperblock 256
#define SHMEM_threads 256 * 4

__global__ void CumulativeSum(int* inputVector, int NumberOfElements)
{
    // Allocate shared memory
    __shared__ int partial_sum[SHMEM_threads];


    // Calculate global thread ID
    int tx = threadIdx.x; // Thread id in a 1D block
    int ty = blockIdx.x;  // Block id in a 1D grid
    int bw = blockDim.x; // Block width

    int tid = tx + ty * bw;

    

    int offset = 1;

    for (int off = 1; off < NumberOfElements; off*=2)
    {
        if (tx > off)
            inputVector[tx] += inputVector[tx - off];

        __syncthreads();
    }


}

int main()
{
    // Define the range for the random numbers (e.g., values between 0 and 99)
    int lowerBound = 0;
    int upperBound = 9;

    // Generate a random list of integers of size NumberOfElements
    //int* randomList = CreateRandomArray(NumberOfElements, lowerBound, upperBound);

    int h_randomList[NumberOfElements] = { 1, 3, 4, 2, 5, 9, 6, 0 };

    printArray(h_randomList, NumberOfElements);
    // Allocation size for all vectors
    size_t bytes = sizeof(int) * NumberOfElements;

    // Device vector pointers
    int* d_randomList;

    

    // Allocate device memory (GPU)
    cudaMalloc(&d_randomList, bytes);

    // Copy to device
    cudaMemcpy(d_randomList, h_randomList, bytes, cudaMemcpyHostToDevice);

    int blocksPerGrid = (NumberOfElements + (threadsperblock - 1)) / threadsperblock;

    // Call kernel
    CumulativeSum << <blocksPerGrid, threadsperblock >> > (d_randomList,NumberOfElements);

    //CumulativeSum << <1, threadsperblock >> > (d_randomList);

    // copy data from device memory to host memory (CPU to  GPU)
    cudaMemcpy(h_randomList, d_randomList, bytes, cudaMemcpyDeviceToHost);

    printArray(h_randomList, NumberOfElements);

    /*
    printArray(randomList, NumberOfElements);
    // Analyze the performance of Counting Sort
    CountingSortAnalysis(randomList, lowerBound, upperBound);
    printArray(randomList, NumberOfElements);
    // Analyze the performance of Radix Sort
    //RadixSortAnalysis(randomList, lowerBound, upperBound)   
    */
;

    // Free the dynamically allocated randomList array

    return 0;
}
