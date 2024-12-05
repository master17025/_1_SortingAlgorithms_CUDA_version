#include "VectorFunc.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <random>

// Function to generate a random array of integers within a given range
int* CreateRandomArray(int NumberOfElements, int lowerBound, int upperBound)
{
    // Create a random number generator using a random device as the seed
    std::random_device rd;  // Provides a seed for the random engine (typically from hardware)
    std::mt19937 gen(rd()); // Mersenne Twister random number engine, generates random numbers
    std::uniform_int_distribution<> dis(lowerBound, upperBound); // Uniform distribution between lowerBound and upperBound

    // Dynamically allocate an array to hold the random numbers
    int* randomList = new int[NumberOfElements];

    // Fill the array with random integers within the specified range
    for (int i = 0; i < NumberOfElements; ++i) {
        randomList[i] = dis(gen); // Generate a random number and assign it to the array
    }

    // Return the pointer to the dynamically allocated array
    return randomList;
}

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