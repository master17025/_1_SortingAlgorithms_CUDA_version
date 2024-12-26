#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <chrono>
#include "RadixSort.cuh"     // CPU Radix Sort implementation


// CPU Merge Function
void merge(std::vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    std::vector<int> L(n1), R(n2);

    for (int i = 0; i < n1; i++) L[i] = arr[left + i];
    for (int j = 0; j < n2; j++) R[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        }
        else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

// CPU Merge Sort
void mergeSortCPU(std::vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        mergeSortCPU(arr, left, mid);
        mergeSortCPU(arr, mid + 1, right);

        merge(arr, left, mid, right);
    }
}

// GPU Merge Sort Kernel (Thrust-based)
std::chrono::duration<double, std::milli> mergeSortGPU(std::vector<int>& arr) {
    
    thrust::device_vector<int> d_arr(arr.begin(), arr.end());

    auto start = std::chrono::high_resolution_clock::now();
    thrust::sort(d_arr.begin(), d_arr.end());
    auto end = std::chrono::high_resolution_clock::now();
    thrust::copy(d_arr.begin(), d_arr.end(), arr.begin());

    

    std::chrono::duration<double, std::milli> duration = end - start;
    return duration;
}
