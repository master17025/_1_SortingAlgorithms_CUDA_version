#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "VectorFunc.h"      // Include custom vector functions
#include "CountingSort.cuh"  // Include CPU Counting Sort implementation
#include "RadixSort.cuh"     // Include CPU Radix Sort implementation
#include "CountingSort.cuh" // GPU Counting Sort
#include "RadixSort.cuh"    // GPU Radix Sort

#include <iostream>
#include <vector>
#include <chrono>  // For measuring execution time
#include <cassert> // For assert

// Function to measure the performance of CPU Counting Sort
void CountingSortAnalysis(std::vector<int>& randomList, int upperBound) {
    auto start = std::chrono::high_resolution_clock::now();
    countingSort(upperBound, randomList.size(), randomList);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Time taken by CPU Counting Sort: " << duration.count() << " ms" << std::endl;
}

// Function to measure the performance of CPU Radix Sort
void RadixSortAnalysis(std::vector<int>& randomList) {
    auto start = std::chrono::high_resolution_clock::now();
    RadixSort(randomList.size(), randomList);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Time taken by CPU Radix Sort: " << duration.count() << " ms" << std::endl;
}

// Function to check if the vector is sorted
bool isSorted(const std::vector<int>& vec) {
    for (size_t i = 1; i < vec.size(); i++) {
        if (vec[i - 1] > vec[i]) {
            return false; // Found an out-of-order pair
        }
    }
    return true; // The vector is sorted
}

int main() {
    // Define the size of the list and bounds
    const long int NumberOfElements = 1 << 4; // 1 million elements
    const int lowerBound = 0;
    const int upperBound = 17; // Range of values [0, 999]

    // Generate a random list of integers
    std::vector<int> randomList(NumberOfElements);
    for (auto& value : randomList) {
        value = rand() % (upperBound - lowerBound + 1) + lowerBound;
    }

    std::cout << "Sorting a list of " << NumberOfElements << " elements." << std::endl;

    // CPU Counting Sort
    printf("------------------------\n");
    std::vector<int> countingSortList = randomList;
    CountingSortAnalysis(countingSortList, upperBound);

    // Check if the vector is sorted
    if (isSorted(countingSortList)) 
        std::cout << "CPU Counting Sort: Array is sorted correctly!" << std::endl;
    
    else 
        std::cerr << "Error: Array not sorted correctly!" << std::endl;
    printf("------------------------\n");
    // GPU Counting Sort
    printf("------------------------\n");
    std::vector<int> countingSortGPUList = randomList;
    CountingSortGPU(countingSortGPUList, upperBound);

    // Check if the vector is sorted
    if (isSorted(countingSortGPUList))
        std::cout << "GPU Counting Sort: Array is sorted correctly!" << std::endl;

    else
        std::cerr << "Error: Array not sorted correctly!" << std::endl;

    printf("------------------------\n");
    // CPU Radix Sort
    printf("------------------------\n");
    std::vector<int> radixSortList = randomList;

    RadixSortAnalysis(radixSortList);

    // Check if the vector is sorted
    if (isSorted(radixSortList))
        std::cout << "CPU Radix Sort: Array is sorted correctly!" << std::endl;

    else
        std::cerr << "Error: Array not sorted correctly!" << std::endl;
    printf("------------------------\n");
    // GPU Radix Sort
    printf("------------------------\n");
    std::vector<int> radixSortGPUList = randomList;
    printVector(radixSortGPUList);
    RadixSortGPU(radixSortGPUList);
    

    // Check if the vector is sorted
    if (isSorted(radixSortGPUList))
        std::cout << "GPU Radix Sort: Array is sorted correctly!" << std::endl;

    else
        std::cerr << "Error: Array not sorted correctly!" << std::endl;

    printf("------------------------\n");
    printVector(radixSortGPUList);

    return 0;
}
//printVector(const std::vector<int>& vec)
