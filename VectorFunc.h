#pragma once
#include <vector>

// Function prototype for creating a random array of integers within a specified range
std::vector<int> CreateRandomArray(int NumberOfElements, int lowerBound, int upperBound);

// Function prototype for printing an array of integers
void printArray(const std::vector<int>& array);
