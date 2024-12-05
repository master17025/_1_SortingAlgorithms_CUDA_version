#pragma once
#include <iostream>
#include <vector>

// Function to perform Radix Sort on the input array
// Radix Sort sorts the array based on individual digits, starting from the least significant digit
// and moving towards the most significant digit.
void RadixSort(int NumberOfElements, std::vector<int>& inputVector);

// Function to perform Counting Sort on the current digit place (used in Radix Sort)
// The `upperBound` parameter defines the current digit place (e.g., ones, tens, hundreds).
// The `NumberOfElements` parameter indicates the number of elements in the input array.
// The `inputVector` is the array that will be sorted based on the current digit.
void countingSortRadix(int upperBound, int NumberOfElements, std::vector<int>& inputVector);
