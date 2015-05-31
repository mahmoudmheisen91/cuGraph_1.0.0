#ifndef MAIN_H_
#define MAIN_H_

#include <stdio.h>
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

__global__ void random_number_generator_kernal(int *masterSeed, int *size, float *PRNG);
__host__ void random_number_generator_host(int *masterSeed, int *itemsPerThread, float *PRNG);
__host__ void grid_block_test_case(int numB, int numT);
__host__ void performance_benchmark(int from, int to);
void sample_run(int size, int num);

#endif

