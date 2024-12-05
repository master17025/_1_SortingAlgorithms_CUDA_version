#include "RadixSort.h"
#include <iostream>
#include <vector>

// Counting sort for radix sort: sorts based on individual digit
void countingSortRadix(int divider, int NumberOfElements, std::vector<int>& inputVector)
{
    // Step 1: Create an output array to store sorted elements
    // Initialize the array with 0s. It has the same size as inputVector.
    std::vector<int> outputArray(NumberOfElements, 0);

    // Step 2: Define the range of digits (0-9) for decimal system
    int range = 10;

    // Step 3: Create a count array to store the frequency of each digit (0-9)
    // Initialize the array with 0s. It has 10 positions for each digit (0 to 9).
    std::vector<int> countArray(range, 0);

    // Step 4: Count the occurrences of each digit at the current digit place (based on divider)
    // The divider will extract the digit at the current place value (1s, 10s, 100s, etc.)
    // The loop goes through all elements of the input array to update the count array.
    for (size_t i = 0; i < NumberOfElements; i++)
    {
        // Extract the digit at the current place value (divider)
        int digit = (inputVector[i] / divider) % range;
        countArray[digit]++; // Increase the count for this digit
    }

    // Step 5: Modify the count array to store the cumulative count
    // This will allow us to place elements in the correct position in outputArray
    for (size_t i = 1; i < range; i++)
    {
        // Add the previous count to the current one to get the cumulative sum
        countArray[i] += countArray[i - 1];
    }

    // Step 6: Build the output array by placing the elements in sorted order
    // We iterate through the input array in reverse order to ensure stability (preserving the order of equal elements)
    for (int i = NumberOfElements - 1; i >= 0; i--)  // Using int for loop to avoid overflow
    {
        // Extract the current digit using the divider
        int digit = (inputVector[i] / divider) % range;

        // Place the element in the correct position in the output array
        outputArray[countArray[digit] - 1] = inputVector[i];

        // Decrease the count for this digit (move to the next position in case of duplicates)
        countArray[digit]--;
    }

    // Step 7: Copy the sorted values from the output array back into the original input array
    // Now inputVector contains the elements sorted based on the current digit.
    inputVector = outputArray;
}

// Main radix sort function: sorts the entire array
void RadixSort(int NumberOfElements, std::vector<int>& inputVector)
{
    // Find the maximum element in the input array to determine the number of digits
    int maximum = *std::max_element(inputVector.begin(), inputVector.end());

    // Step 1: Call countingSortRadix for each digit place (1s, 10s, 100s, etc.)
    // We start from the least significant digit and move to the more significant ones
    for (size_t divider = 1; maximum / divider > 0; divider *= 10)
    {
        // Perform counting sort based on the current digit
        countingSortRadix(divider, NumberOfElements, inputVector);
    }
}
