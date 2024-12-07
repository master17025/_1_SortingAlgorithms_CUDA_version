#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "VectorFunc.h"    // Include custom array functions (e.g., CreateRandomArray)
#include "CountingSort.cuh"  // Include Counting Sort implementation
#include "RadixSort.h"     // Include Radix Sort implementation

#include <iostream>
#include <chrono>  // For measuring execution time
// Define the number of elements of the integer array
int const NumberOfElements = 20; // Example: 450 million elements

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

#define threadsperblock 1024

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





int main()
{
    // Define the range for the random numbers (e.g., values between 0 and 99)
    int lowerBound = 1;
    int upperBound = 99;

    // Generate a random list of integers of size NumberOfElements
    int* h_randomList = CreateRandomArray(NumberOfElements, lowerBound, upperBound);
    
    CumulativeSumAnalysis(h_randomList, lowerBound, upperBound);
    

    h_randomList = CreateRandomArray(NumberOfElements, lowerBound, upperBound);
    // Allocation size for all vectors
   CountingSortAnalysis(h_randomList, lowerBound, upperBound);


    //h_randomList = CreateRandomArray(NumberOfElements, lowerBound, upperBound);
    //RadixSortAnalysis(h_randomList, lowerBound, upperBound);


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
