#pragma once
#include <iostream>
#include <vector>

// Function to merge two sorted subarrays into one sorted subarray
// This function merges the two sorted parts of the array (arr[left..middle] and arr[middle+1..right])
// into a single sorted array.
void Merge(int left, int middle, int right, std::vector<int>& arr);

// Function to perform Merge Sort on the array
// This is the main recursive function that divides the array into two halves and then
// calls Merge to combine the sorted halves back together.
// 'left' is the starting index of the array/subarray to be sorted.
// 'right' is the ending index of the array/subarray to be sorted.
void MergeSort(int left, int right, std::vector<int>& arr);
