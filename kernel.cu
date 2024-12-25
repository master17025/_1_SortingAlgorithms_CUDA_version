#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "VectorFunc.h"      // Custom vector functions
#include "CountingSort.cuh"  // CPU Counting Sort implementation
#include "RadixSort.cuh"     // CPU Radix Sort implementation

#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>

// Function to check if the vector is sorted
bool isSorted(const std::vector<int>& vec) {
    for (size_t i = 1; i < vec.size(); i++) {
        if (vec[i - 1] > vec[i]) {
            return false; // Found an out-of-order pair
        }
    }
    return true; // The vector is sorted
}

// Function to compare CPU and GPU Counting Sort
void CompareCountingSort(const std::vector<int>& inputList, int upperBound) {
    std::vector<int> cpuList = inputList;
    std::vector<int> gpuList = inputList;

    // CPU Counting Sort
    auto startCPU = std::chrono::high_resolution_clock::now();
    countingSort(upperBound, cpuList.size(), cpuList);
    auto endCPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> durationCPU = endCPU - startCPU;

    // GPU Counting Sort
    //printVector(gpuList);
    std::chrono::duration<double, std::milli> durationGPU  = CountingSortGPU(gpuList, upperBound);


    // Compare Results
    std::cout << "CPU Counting Sort Time: " << durationCPU.count() << " ms" << std::endl;
    std::cout << "GPU Counting Sort Time: " << durationGPU.count() << " ms" << std::endl;
    //printVector(gpuList);
    // Check correctness
    if (isSorted(cpuList) && isSorted(gpuList) && cpuList == gpuList) {
        std::cout << "Counting Sort: Both CPU and GPU results are correct!" << std::endl;
    }
    else {
        std::cerr << "Error: Counting Sort results do not match or are incorrect!" << std::endl;
    }
    printf("------------------------\n");
}

// Function to compare CPU and GPU Radix Sort
void CompareRadixSort(const std::vector<int>& inputList) {
    std::vector<int> cpuList = inputList;
    std::vector<int> gpuList = inputList;

    // CPU Radix Sort
    auto startCPU = std::chrono::high_resolution_clock::now();
    RadixSort(cpuList.size(), cpuList);
    auto endCPU = std::chrono::high_resolution_clock::now();
    printVector(cpuList);
    std::chrono::duration<double, std::milli> durationCPU = endCPU - startCPU;

    // GPU Radix Sort
    //printVector(gpuList);
    std::chrono::duration<double, std::milli> durationGPU  = RadixSortGPU(gpuList);
    printVector(gpuList);
    // Compare Results
    std::cout << "CPU Radix Sort Time: " << durationCPU.count() << " ms" << std::endl;
    std::cout << "GPU Radix Sort Time: " << durationGPU.count() << " ms" << std::endl;

    

    // Check correctness
    if (isSorted(cpuList) && isSorted(gpuList) && cpuList == gpuList) {
        std::cout << "Radix Sort: Both CPU and GPU results are correct!" << std::endl;
    }
    else {
        std::cerr << "Error: Radix Sort results do not match or are incorrect!" << std::endl;
    }
    printf("------------------------\n");
}

int main() {
    // Define the size of the list and bounds
    const long int NumberOfElements = 1 << 6; // 1 million elements
    const int lowerBound = 0;
    const int upperBound = 1000; // Range of values [0, 999]

    // Generate a random list of integers
    std::vector<int> randomList(NumberOfElements);
    for (auto& value : randomList) {
        value = rand() % (upperBound - lowerBound + 1) + lowerBound;
    }

    std::cout << "Sorting a list of " << NumberOfElements << " elements." << std::endl;

    // Compare CPU and GPU Counting Sort
    printf("***** Comparing Counting Sort *****\n");
    CompareCountingSort(randomList, upperBound);

    // Compare CPU and GPU Radix Sort
    printf("***** Comparing Radix Sort *****\n");
    CompareRadixSort(randomList);

    return 0;
}
