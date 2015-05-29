#ifndef MAIN_H_
#define MAIN_H_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <cuda.h>

#define N 8
#define gridSize 1
#define blockSize N

using namespace std;

__global__ void random_number_generator_kernal(int *masterSeed, int *itemsPerThread, int *PRNG);

#endif

