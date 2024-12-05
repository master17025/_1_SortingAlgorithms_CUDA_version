
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "VectorFunc.h"    // Include custom vector functions (e.g., CreateRandomArray)
#include "MergeSort.h"     // Include Merge Sort implementation
#include "CountingSort.h"  // Include Counting Sort implementation
#include "RadixSort.h"     // Include Radix Sort implementation

#include <iostream>
#include <vector>
#include <chrono>  // For measuring execution time


// Define the number of elements of the integer vector
int NumberOfElements = 4.5e8; //30

// int NumberOfElements = 17.6e6;
// Time taken to sort the list using Merge sort : 31082.6 milliseconds

// int NumberOfElements = 11e8;
// Time taken to sort the list using Counting sort : 32402.2 milliseconds

// int NumberOfElements = 4.5e8;
//Time taken to sort the list using Radix sort: 30693.6 milliseconds


// Function to measure the time and performance of Merge Sort
void MergeSortAnalasis(std::vector<int> randomList, int lowerBound, int upperBound)
{

    // Set the indices for the Merge Sort algorithm (left and right)
    int left = 0;
    int right = randomList.size() - 1;
    // Start the timer to measure execution time
    auto start = std::chrono::high_resolution_clock::now();

    // Perform Merge Sort on the random list
    MergeSort(left, right, randomList);

    // Stop the timer after sorting is complete
    auto end = std::chrono::high_resolution_clock::now();
    // Calculate the time duration in milliseconds
    std::chrono::duration<double, std::milli> duration = end - start;

    // Output the sorted list (optional, to verify correctness)
    std::cout << "Sorted list:" << std::endl;
    // std::cout << randomList; // Uncomment this if you want to see the sorted list

    // Output the time taken to sort using Merge Sort
    std::cout << "Time taken to sort the list using Merge sort: " << duration.count() << " milliseconds" << std::endl;

}


// Function to measure the time and performance of Counting Sort
void CountingSortAnalysis(std::vector<int> randomList, int lowerBound, int upperBound)
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
void RadixSortAnalysis(std::vector<int> randomList, int lowerBound, int upperBound)
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



int main()
{
    // Define the range for the random numbers (e.g., values between 0 and 99)
    int lowerBound = 0;
    int upperBound = 99;

    // Generate a random list of integers of size NumberOfElements
    std::vector<int> randomList = CreateRandomArray(NumberOfElements, lowerBound, upperBound);


    // Uncomment one of these functions to test the sorting algorithm of your choice:

    // Analyze the performance of Merge Sort
    //MergeSortAnalasis(randomList, lowerBound, upperBound);
    std::cout << "--------------------------------" << std::endl;

    // Analyze the performance of Counting Sort
    //CountingSortAnalysis(randomList, lowerBound, upperBound);
    std::cout << "--------------------------------" << std::endl;

    // Analyze the performance of Radix Sort
    RadixSortAnalysis(randomList, lowerBound, upperBound);
    std::cout << "--------------------------------" << std::endl;
    return 0;
}