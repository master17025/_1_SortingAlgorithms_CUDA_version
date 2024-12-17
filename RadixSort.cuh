#pragma once
#include <iostream>
#include <algorithm> // For std::max_element

// Function to perform Radix Sort on the input array
// Radix Sort sorts the array based on individual digits, starting from the least significant digit
// and moving towards the most significant digit.
void RadixSort(long int NumberOfElements, int* inputArray);
// Function to perform Counting Sort on the current digit place (used in Radix Sort)
// The `upperBound` parameter defines the current digit place (e.g., ones, tens, hundreds).
// The `NumberOfElements` parameter indicates the number of elements in the input array.
// The `inputVector` is the array that will be sorted based on the current digit.
void countingSortRadix(int divider, long int NumberOfElements, int* inputArray);

__global__ void CountDigitsKernel(int* input, int* count, int size, int divider, int range);
__global__ void PlaceElementsKernel(int* input, int* output, int* count, int size, int divider, int range);

void RadixSortGPU(int* h_input, int size);