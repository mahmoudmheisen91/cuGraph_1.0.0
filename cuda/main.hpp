#ifndef MAIN_H_
#define MAIN_H_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <cuda.h>
#include <ctime>
#include <algorithm>

#define N 1024
#define gridSize 1
#define blockSize N

using namespace std;

__global__ void random_number_generator_kernal(int *masterSeed, int *itemsPerThread, float *PRNG);


#endif

