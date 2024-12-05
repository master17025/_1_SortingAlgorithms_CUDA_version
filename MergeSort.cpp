#include "MergeSort.h"
#include <iostream>
#include <vector>

// Merge function merges two sorted subarrays into a single sorted array
// This function is called by MergeSort to combine the two halves of the array.
// It assumes the left part arr[left..middle] and the right part arr[middle+1..right]
// are already sorted, and it merges them into arr[left..right] in sorted order.
void Merge(int left, int middle, int right, std::vector<int>& arr)
{
    // Calculate the sizes of the two subarrays to be merged
    int n1 = middle - left + 1;
    int n2 = right - middle;

    // Create temporary arrays to hold the data from the two subarrays
    std::vector<int> L(n1);  // Left subarray
    std::vector<int> R(n2);  // Right subarray

    // Copy data into the temporary subarrays
    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];  // Copy left half of the array
    for (int i = 0; i < n2; i++)
        R[i] = arr[middle + 1 + i];  // Copy right half of the array

    // Merge the temporary subarrays back into the original array in sorted order
    int i = 0, j = 0, k = left;

    // Compare elements from both subarrays and place the smaller element into arr[k]
    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])  // If L[i] is smaller or equal, copy it to arr[k]
        {
            arr[k] = L[i];
            i++;
        }
        else  // If R[j] is smaller, copy it to arr[k]
        {
            arr[k] = R[j];
            j++;
        }
        k++;  // Move to the next position in the original array
    }

    // If there are remaining elements in L, copy them into arr
    while (i < n1)
    {
        arr[k] = L[i];
        i++;
        k++;
    }

    // If there are remaining elements in R, copy them into arr
    while (j < n2)
    {
        arr[k] = R[j];
        j++;
        k++;
    }
}

// MergeSort function recursively divides the array into two halves, 
// sorts each half, and then merges the sorted halves back together.
void MergeSort(int left, int right, std::vector<int>& arr)
{
    if (left < right)  // Base case: when left equals right, the array has only one element
    {
        // Find the middle index of the array
        int middle = left + (right - left) / 2;

        // Recursively sort the left and right halves
        MergeSort(left, middle, arr);
        MergeSort(middle + 1, right, arr);

        // Merge the two sorted halves
        Merge(left, middle, right, arr);
    }
}
