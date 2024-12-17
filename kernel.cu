#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "VectorFunc.h"    // Include custom array functions (e.g., CreateRandomArray)
#include "CountingSort.cuh"  // Include Counting Sort implementation
#include "RadixSort.cuh" // Include Radix Sort implementation
#include <cassert> // For assert

#include <iostream>
#include <chrono>  // For measuring execution time


// Function to measure the time and performance of Counting Sort
void CountingSortAnalysis(std::vector<int>& randomList, int lowerBound, int upperBound) {
    auto start = std::chrono::high_resolution_clock::now();
    countingSort(upperBound, randomList.size(), randomList);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Time taken to sort the list using Counting Sort: " << duration.count() << " milliseconds" << std::endl;
}

// Function to measure the time and performance of Radix Sort
void RadixSortAnalysis(std::vector<int>& randomList) {
    auto start = std::chrono::high_resolution_clock::now();
    RadixSort(randomList.size(), randomList);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Time taken to sort the list using Radix Sort: " << duration.count() << " milliseconds" << std::endl;
}



int main() 
{
    // Define the size of the list and bounds
    const long int NumberOfElements = 1e7; // 1 million elements
    const int lowerBound = 0;
    const int upperBound = 999; // Range of values [0, 999]

    // Generate a random list of integers within the specified bounds
    std::vector<int> randomList(NumberOfElements);
    for (auto& value : randomList) {
        value = rand() % (upperBound - lowerBound + 1) + lowerBound;
    }

    std::cout << "Sorting a list of " << NumberOfElements << " elements." << std::endl;

    // Perform Counting Sort Analysis
    std::vector<int> countingSortList = randomList; // Make a copy for Counting Sort
    CountingSortAnalysis(countingSortList, lowerBound, upperBound);

    // Perform Radix Sort Analysis
    std::vector<int> radixSortList = randomList; // Make a copy for Radix Sort
    RadixSortAnalysis(radixSortList);

    return 0;
}
