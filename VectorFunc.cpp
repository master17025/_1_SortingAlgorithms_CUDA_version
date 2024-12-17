#include <iostream>
#include <vector>

// Function to print the contents of an integer vector
void printVector(const std::vector<int>& vec)
{
    // Output a label indicating we're about to print the vector
    std::cout << "Vector: ";

    // Iterate through each element in the vector and print it
    for (const int& element : vec) {
        std::cout << element << " ";  // Print each number followed by a space
    }

    // Print a newline after the vector for formatting
    std::cout << std::endl;
}
