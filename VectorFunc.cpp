#include "VectorFunc.h"
#include <iostream>
#include <random>
#include <vector>

// Function to generate a random array of integers within a given range
std::vector<int> CreateRandomArray(int NumberOfElements, int lowerBound, int upperBound)
{
    // Create a random number generator using a random device as the seed
    std::random_device rd;  // Provides a seed for the random engine (typically from hardware)
    std::mt19937 gen(rd()); // Mersenne Twister random number engine, generates random numbers
    std::uniform_int_distribution<> dis(lowerBound, upperBound); // Uniform distribution between lowerBound and upperBound

    // Create a vector to hold the random numbers
    std::vector<int> randomList;

    // Fill the vector with random integers within the specified range
    for (int i = 0; i < NumberOfElements; ++i) {
        randomList.push_back(dis(gen)); // Generate a random number and add it to the list
    }

    // Return the generated list of random numbers
    return randomList;
}

// Function to print the contents of an integer array (vector)
void printArray(const std::vector<int>& array)
{
    // Output a label indicating we're about to print the array
    std::cout << "Array: ";

    // Iterate through each element in the vector and print it
    for (int num : array) {
        std::cout << num << " ";  // Print each number followed by a space
    }

    // Print a newline after the array for formatting
    std::cout << std::endl;
}