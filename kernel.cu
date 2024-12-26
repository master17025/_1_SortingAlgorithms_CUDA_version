#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "VectorFunc.h"      // Custom vector functions
#include "CountingSort.cuh"  // CPU Counting Sort implementation
#include "RadixSort.cuh"     // CPU Radix Sort implementation

#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>
#include <fstream>
#include <numeric>
// Function to check if the vector is sorted
bool isSorted(const std::vector<int>& vec) {
    for (size_t i = 1; i < vec.size(); i++) {
        if (vec[i - 1] > vec[i]) {
            return false; // Found an out-of-order pair
        }
    }
    return true; // The vector is sorted
}

// Function to write results to CSV file
void writeToCSV(const std::string& filename, const std::vector<std::tuple<int, double, double>>& data) {
    std::ofstream file(filename);
    file << "Number of Elements,CPU Time (ms),GPU Time (ms)\n";
    for (const auto& entry : data) {
        file << std::get<0>(entry) << "," << std::get<1>(entry) << "," << std::get<2>(entry) << "\n";
    }
    file.close();
}

// Function to compare CPU and GPU Counting Sort
void CompareCountingSort(const std::vector<int>& inputList, int upperBound, std::vector<std::tuple<int, double, double>>& results) {
    std::vector<int> cpuList = inputList;
    std::vector<int> gpuList = inputList;

    std::vector<double> cpuTimes;
    std::vector<double> gpuTimes;

    for (int i = 0; i < 5; ++i) {
        cpuList = inputList;
        gpuList = inputList;

        // CPU Counting Sort
        auto startCPU = std::chrono::high_resolution_clock::now();
        countingSort(upperBound, cpuList.size(), cpuList);
        auto endCPU = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> durationCPU = endCPU - startCPU;
        cpuTimes.push_back(durationCPU.count());

        // GPU Counting Sort
        std::chrono::duration<double, std::milli> durationGPU = CountingSortGPU(gpuList, upperBound);
        gpuTimes.push_back(durationGPU.count());
    }

    double avgCpuTime = std::accumulate(cpuTimes.begin(), cpuTimes.end(), 0.0) / cpuTimes.size();
    double avgGpuTime = std::accumulate(gpuTimes.begin(), gpuTimes.end(), 0.0) / gpuTimes.size();

    // Store results
    results.emplace_back(inputList.size(), avgCpuTime, avgGpuTime);

    // Compare Results
    std::cout << "Average CPU Counting Sort Time: " << avgCpuTime << " ms" << std::endl;
    std::cout << "Average GPU Counting Sort Time: " << avgGpuTime << " ms" << std::endl;

    // Check correctness
    if (isSorted(cpuList) && isSorted(gpuList) && cpuList == gpuList) {
        std::cout << "Counting Sort: Both CPU and GPU results are correct!" << std::endl;
    }
    else {
        std::cerr << "Error: Counting Sort results do not match or are incorrect!" << std::endl;
    }
    printf("------------------------\n");
}

// Function to compare CPU and GPU Merge Sort
void CompareMergeSort(const std::vector<int>& inputList, std::vector<std::tuple<int, double, double>>& results) {
    std::vector<int> cpuList = inputList;
    std::vector<int> gpuList = inputList;

    std::vector<double> cpuTimes;
    std::vector<double> gpuTimes;

    for (int i = 0; i < 5; ++i) {
        cpuList = inputList;
        gpuList = inputList;

        // CPU Merge Sort
        auto startCPU = std::chrono::high_resolution_clock::now();
        mergeSortCPU(cpuList, 0, cpuList.size() - 1);
        auto endCPU = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> durationCPU = endCPU - startCPU;
        cpuTimes.push_back(durationCPU.count());

        // GPU Merge Sort
        std::chrono::duration<double, std::milli> durationGPU = mergeSortGPU(gpuList);
        gpuTimes.push_back(durationGPU.count());
    }

    double avgCpuTime = std::accumulate(cpuTimes.begin(), cpuTimes.end(), 0.0) / cpuTimes.size();
    double avgGpuTime = std::accumulate(gpuTimes.begin(), gpuTimes.end(), 0.0) / gpuTimes.size();

    // Store results
    results.emplace_back(inputList.size(), avgCpuTime, avgGpuTime);

    // Compare Results
    std::cout << "Average CPU Merge Sort Time: " << avgCpuTime << " ms" << std::endl;
    std::cout << "Average GPU Merge Sort Time: " << avgGpuTime << " ms" << std::endl;

    // Check correctness
    if (isSorted(cpuList) && isSorted(gpuList) && cpuList == gpuList) {
        std::cout << "Merge Sort: Both CPU and GPU results are correct!" << std::endl;
    }
    else {
        std::cerr << "Error: Merge Sort results do not match or are incorrect!" << std::endl;
    }
    printf("------------------------\n");
}

int main() {
    const int lowerBound = 0;
    const int upperBound = 100;

    // Vectors to store results
    std::vector<std::tuple<int, double, double>> countingSortResults;
    std::vector<std::tuple<int, double, double>> mergeSortResults;

    for (int exp = 4; exp <= 28; ++exp) {
        long int NumberOfElements = 1 << exp;

        // Generate a random list of integers
        std::vector<int> randomList(NumberOfElements);
        for (auto& value : randomList) {
            value = rand() % (upperBound - lowerBound + 1) + lowerBound;
        }

        std::cout << "Sorting a list of " << NumberOfElements << " elements." << std::endl;

        // Compare CPU and GPU Counting Sort
        printf("***** Comparing Counting Sort *****\n");
        CompareCountingSort(randomList, upperBound, countingSortResults);

        // Compare CPU and GPU Merge Sort
        printf("***** Comparing Merge Sort *****\n");
        CompareMergeSort(randomList, mergeSortResults);
    }

    // Write results to CSV files
    writeToCSV("counting_sort.csv", countingSortResults);
    writeToCSV("merge_sort.csv", mergeSortResults);

    return 0;
}