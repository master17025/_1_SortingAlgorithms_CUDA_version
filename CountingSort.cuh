#pragma once
#include <iostream>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// Function to perform Counting Sort on the input array
// The function works by counting the occurrences of each element
// in a range (determined by the upperBound) and then placing the elements
// in the correct order in the inputVector.
void countingSort(int upperBound, long int NumberOfElements, std::vector<int>& inputArray);

void CountingSortGPU(std::vector<int>& h_input, int upperBound);