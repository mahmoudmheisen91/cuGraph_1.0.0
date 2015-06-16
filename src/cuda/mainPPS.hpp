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

//#define NUM_BANKS 16
//#define LOG_NUM_BANKS 4

//#define CONFLICT_FREE_OFFSET(n) \
//	((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS)) 

using namespace std;

__device__ void cuScanBlock(int *g_odata, int *g_idata, int n);
__global__ void scanGlobal(int *g_odata, int *g_idata, int n);
__host__ void inclusive_scan_sum(int *array, int length);
__host__ void exclusive_scan_sum(int *array, int length);
__host__ void printArray(int *array, int length);
void test_scan(int size);

#endif // MAINPPS_HPP_

