#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "VectorFunc.h"    // Include custom array functions (e.g., CreateRandomArray)
#include "CountingSort.cuh"  // Include Counting Sort implementation
#include "RadixSort.cuh" // Include Radix Sort implementation
#include"RandomGeneration.cuh"
#include <cassert> // For assert

#include <iostream>
#include <chrono>  // For measuring execution time


// Function to measure the time and performance of Counting Sort
void CountingSortAnalysis(int* randomList, int lowerBound, int upperBound, long int NumberOfElements) {
    auto start = std::chrono::high_resolution_clock::now();
    countingSort(upperBound, NumberOfElements, randomList);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Time taken to sort the list using Counting sort: " << duration.count() << " milliseconds" << std::endl;
}

// Function to measure the time and performance of Radix Sort
void RadixSortAnalysis(int* randomList, int lowerBound, int upperBound, long int NumberOfElements) {
    auto start = std::chrono::high_resolution_clock::now();
    RadixSort(NumberOfElements, randomList);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Time taken to sort the list using Radix sort: " << duration.count() << " milliseconds" << std::endl;
}


int main() {

    // Define the number of elements of the integer array
    long int NumberOfElements = 20e8; // Example: 450 million elements

    int lowerBound = 1;
    int upperBound = 9;
     
    // Generate random array on GPU
    int* h_randomList = CreateRandomArray(NumberOfElements, lowerBound, upperBound);

    // Perform analysis

    CountingSortAnalysis(h_randomList, lowerBound, upperBound,NumberOfElements);

    //RadixSortAnalysis(h_randomList, lowerBound, upperBound,NumberOfElements);

    CountingSortGPU(upperBound, NumberOfElements, h_randomList);

    // Assertion to verify the array is sorted
    for (long int i = 1; i < NumberOfElements; ++i) {
        assert(inputArray[i - 1] <= inputArray[i] && "Array is not sorted!");
    }

    std::cout << "Array is sorted correctly!" << std::endl;


    // Free host memory
    delete[] h_randomList;

    return 0;
}
