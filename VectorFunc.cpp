#include "VectorFunc.h"



// Function to print the contents of an integer array
void printArray(const int* array, int size)
{
    // Output a label indicating we're about to print the array
    std::cout << "Array: ";

    // Iterate through each element in the array and print it
    for (int i = 0; i < size; ++i) {
        std::cout << array[i] << " ";  // Print each number followed by a space
    }

    // Print a newline after the array for formatting
    std::cout << std::endl;
}