#include "CountingSort.h"
#include <iostream>

void countingSort(int upperBound, int NumberOfElements, int* inputArray)
{
    // Step 1: The inputArray has `NumberOfElements` elements to sort,
    // and `upperBound` specifies the maximum value in the array.

    // Step 2: Create an output array to store the sorted elements.
    // Dynamically allocate memory for the output array.
    int* outputArray = new int[NumberOfElements]();

    // Step 3: Define the range of possible values.
    int range = upperBound + 1;

    // Step 4: Create a count array to store the frequency of each element.
    // Dynamically allocate memory for the count array and initialize to 0.
    int* countArray = new int[range]();

    // Step 5: Count the occurrences of each element in the inputArray.
    for (int i = 0; i < NumberOfElements; ++i)
        ++countArray[inputArray[i]];

    // Step 6: Modify the count array to store cumulative counts.
    for (int i = 1; i < range; ++i)
        countArray[i] += countArray[i - 1];

    // Step 7: Place elements from the input array into the output array using the count array.
    for (int i = NumberOfElements - 1; i >= 0; --i) {
        outputArray[--countArray[inputArray[i]]] = inputArray[i];
    }

    // Step 8: Transfer the sorted values from the output array back to the input array.
    for (int i = 0; i < NumberOfElements; ++i)
        inputArray[i] = outputArray[i];

    // Step 9: Free the dynamically allocated memory.
    delete[] outputArray;
    delete[] countArray;
}
