#include "RadixSort.h"


// Counting sort for radix sort: sorts based on individual digit
void countingSortRadix(int divider, int NumberOfElements, int* inputArray)
{
    // Step 1: Create an output array to store sorted elements
    int* outputArray = new int[NumberOfElements]();

    // Step 2: Define the range of digits (0-9) for the decimal system
    int range = 10;

    // Step 3: Create a count array to store the frequency of each digit (0-9)
    int* countArray = new int[range]();

    // Step 4: Count the occurrences of each digit at the current digit place (based on divider)
    for (int i = 0; i < NumberOfElements; i++)
    {
        int digit = (inputArray[i] / divider) % range;
        countArray[digit]++;
    }

    // Step 5: Modify the count array to store the cumulative count
    for (int i = 1; i < range; i++)
    {
        countArray[i] += countArray[i - 1];
    }

    // Step 6: Build the output array by placing the elements in sorted order
    for (int i = NumberOfElements - 1; i >= 0; i--)
    {
        int digit = (inputArray[i] / divider) % range;
        outputArray[countArray[digit] - 1] = inputArray[i];
        countArray[digit]--;
    }

    // Step 7: Copy the sorted values from the output array back into the original input array
    for (int i = 0; i < NumberOfElements; i++)
    {
        inputArray[i] = outputArray[i];
    }

    // Step 8: Free dynamically allocated memory
    delete[] outputArray;
    delete[] countArray;
}

// Main radix sort function: sorts the entire array
void RadixSort(int NumberOfElements, int* inputArray)
{
    // Find the maximum element in the input array to determine the number of digits
    int maximum = *std::max_element(inputArray, inputArray + NumberOfElements);

    // Step 1: Call countingSortRadix for each digit place (1s, 10s, 100s, etc.)
    for (int divider = 1; maximum / divider > 0; divider *= 10)
    {
        countingSortRadix(divider, NumberOfElements, inputArray);
    }
}
