#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <chrono>

void merge(std::vector<int>& arr, int left, int mid, int right);
void mergeSortCPU(std::vector<int>& arr, int left, int right);

std::chrono::duration<double, std::milli> mergeSortGPU(std::vector<int>& arr);