#pragma once
#include <iostream>
#include <vector>

// Function to perform Counting Sort on the input array
// The function works by counting the occurrences of each element
// in a range (determined by the upperBound) and then placing the elements
// in the correct order in the inputVector.
void countingSort(int upperBound, int NumberOfElements, int* inputArray);
