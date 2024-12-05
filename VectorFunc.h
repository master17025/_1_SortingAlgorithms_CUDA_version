#pragma once
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


// Function prototype for creating a random array of integers within a specified range
int* CreateRandomArray(int NumberOfElements, int lowerBound, int upperBound);

// Function prototype for printing an array of integers
void printArray(const int* array, int size);