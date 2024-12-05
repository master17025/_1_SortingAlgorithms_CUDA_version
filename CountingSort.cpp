#include "CountingSort.h"
#include <iostream>
#include <vector>

void countingSort(int upperBound, int NumberOfElements, std::vector<int>& inputVector)
{
    // Step 1: We are given the upper bound (the maximum value in the input array)
    // and the number of elements to be sorted in inputVector.

    // Step 2: Create an output array where the sorted elements will be placed.
    // The output array is initialized with zeros, and it will store the sorted values.
    std::vector<int> outputArray(NumberOfElements, 0);

    // Step 3: Define the range of possible values.
    // Since counting sort works with integer values, the range will be from 0 to upperBound (inclusive).
    int range = upperBound + 1;

    // Step 4: Create a count array to store the frequency of each element.
    // The size of the count array is `range`, and it is initialized with zero.
    std::vector<int> countArray(range, 0);

    // Step 5: Count the occurrences of each element in the inputVector.
    // For each element in inputVector, increment the corresponding position in the count array.
    // This step counts how many times each integer appears in inputVector.
    for (size_t i = 0; i < NumberOfElements; i++) // parallelizable 1D
        ++countArray[inputVector[i]]; // Increment count at the index of the value in inputVector

    // Step 6: Modify the count array to store cumulative counts.
    // This step converts the count array to store the number of elements less than or equal to each index.
    for (size_t i = 1; i < range; i++) // parallelizable 2D
        countArray[i] += countArray[i - 1];

    // Step 7: Place elements from the input array to the output array using the count array.
    // We process each element from the input array and place it in the correct position in outputArray.
    // The count array is used to determine the position of each element.
    // We decrement the corresponding count in countArray to avoid overwriting elements.
    for (size_t i = 0; i < NumberOfElements; i++) // parallelizable 1D ?
        outputArray[--countArray[inputVector[i]]] = inputVector[i]; // Decrement count and place element in output

    // Step 8: Transfer the sorted values from outputArray back to inputVector.
    // After sorting, the inputVector will contain the sorted array.
    inputVector = outputArray;
}
