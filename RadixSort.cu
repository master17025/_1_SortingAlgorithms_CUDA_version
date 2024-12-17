#include "RadixSort.cuh"

#include <iostream>
#include <chrono>  // For measuring execution time

#include <cub/cub.cuh>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// Counting sort for radix sort: sorts based on individual digit
void countingSortRadix(int divider, long int NumberOfElements, int* inputArray)
{
    // Step 1: Create an output array to store sorted elements
    int* outputArray = new int[NumberOfElements]();

    // Step 2: Define the range of digits (0-9) for the decimal system
    int range = 10;

    // Step 3: Create a count array to store the frequency of each digit (0-9)
    int* countArray = new int[range]();

    // Step 4: Count the occurrences of each digit at the current digit place (based on divider)
    for (long int i = 0; i < NumberOfElements; i++)
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
    for (long int i = NumberOfElements - 1; i >= 0; i--)
    {
        int digit = (inputArray[i] / divider) % range;
        outputArray[countArray[digit] - 1] = inputArray[i];
        countArray[digit]--;
    }

    // Step 7: Copy the sorted values from the output array back into the original input array
    for (long int i = 0; i < NumberOfElements; i++)
    {
        inputArray[i] = outputArray[i];
    }

    // Step 8: Free dynamically allocated memory
    delete[] outputArray;
    delete[] countArray;
}

// Main radix sort function: sorts the entire array
void RadixSort(long int NumberOfElements, int* inputArray)
{
    // Find the maximum element in the input array to determine the number of digits
    int maximum = *std::max_element(inputArray, inputArray + NumberOfElements);

    // Step 1: Call countingSortRadix for each digit place (1s, 10s, 100s, etc.)
    for (int divider = 1; maximum / divider > 0; divider *= 10)
    {
        countingSortRadix(divider, NumberOfElements, inputArray);
    }
}
#define threadsperblock 1024

// Kernel pour compter les occurrences des chiffres
__global__ void CountDigitsKernel(int* input, int* count, int size, int divider, int range) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        int digit = (input[tid] / divider) % range;
        atomicAdd(&count[digit], 1); // Incrément atomique pour éviter les conflits
    }
}

// Kernel pour placer les éléments triés dans le tableau de sortie
__global__ void PlaceElementsKernel(int* input, int* output, int* count, int size, int divider, int range) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        int digit = (input[tid] / divider) % range;
        int position = atomicSub(&count[digit], 1) - 1;
        output[position] = input[tid];
    }
}

// Fonction principale Radix Sort sur GPU
void RadixSortGPU(int* h_input, int size) {
    const int range = 10;

    // Allocation mémoire GPU
    int* d_input, * d_output, * d_count;
    cudaMalloc(&d_input, size * sizeof(int));
    cudaMalloc(&d_output, size * sizeof(int));
    cudaMalloc(&d_count, range * sizeof(int));
    // Trouver le maximum pour connaître le nombre de chiffres
    int maxElement = *std::max_element(h_input, h_input + size);

    auto start = std::chrono::high_resolution_clock::now();
    // Copie du tableau d'entrée vers le GPU
    cudaMemcpy(d_input, h_input, size * sizeof(int), cudaMemcpyHostToDevice);

    // Boucle sur chaque position décimale
    for (int divider = 1; maxElement / divider > 0; divider *= 10) {

        cudaMemset(d_count, 0, range * sizeof(int));

        // Compter les occurrences des chiffres en parallèle
        int blocks = (size + threadsperblock - 1) / threadsperblock;
        CountDigitsKernel << <blocks, threadsperblock >> > (d_input, d_count, size, divider, range);
        cudaDeviceSynchronize();

        // Calculer les sommes cumulatives
        int h_count[range] = { 0 };
        cudaMemcpy(h_count, d_count, range * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 1; i < range; i++) {
            h_count[i] += h_count[i - 1];
        }
        cudaMemcpy(d_count, h_count, range * sizeof(int), cudaMemcpyHostToDevice);

        // Placer les éléments dans le tableau de sortie en parallèle
        PlaceElementsKernel << <blocks, threadsperblock >> > (d_input, d_output, d_count, size, divider, range);
        cudaDeviceSynchronize();

        // Échange des pointeurs pour préparer l'itération suivante
        std::swap(d_input, d_output);
    }



    // Copier les résultats triés vers l'hôte
    cudaMemcpy(h_input, d_input, size * sizeof(int), cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Time taken to sort using Rdix Sort on GPU: " << duration.count() << " ms" << std::endl;

    // Libérer la mémoire GPU
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_count);
}