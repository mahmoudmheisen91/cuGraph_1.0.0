#ifndef MAINPPS_HPP_
#define MAINPPS_HPP_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <cuda.h>
#include <ctime>
#include <algorithm>
#include <math.h>

#define N pow(2, 10)
#define B pow(2, 16)
#define gridSize B
#define blockSize N
	
using namespace std;

__global__ void cuScan(int *g_odata, int *g_idata, int n);
__host__ void inclusive_scan_sum(int *array, int length);
__host__ void exclusive_scan_sum(int *array, int length);
__host__ void printArray(int *array, int length);
void test_scan(int size, int num);

#endif // MAINPPS_HPP_
