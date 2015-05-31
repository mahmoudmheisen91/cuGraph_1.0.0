#include "main.hpp"

__global__ void random_number_generator_kernal(int *masterSeed, int *size, float *PRNG) {
	long int a = 16807;                      // same as apple c++ imp
	long int m = 2147483647;                 // 2^31 − 1
	float rec  = 1.0 / m;
	
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	//int y = threadIdx.y + blockIdx.y * blockDim.y;
	//int tid = x + y * blockDim.x * gridDim.x;

	long int seed = *masterSeed + tid;
	
	long int theta, temp;
	while (tid < *size) {
		temp = seed * a;                       // seed = Xn , c = 0
		theta = temp - m * floor(temp * rec);  // is the same as (temp mod m) ((Xn * a) mod m)
		PRNG[tid] = (float)theta/m;			   // between 1/m - 1
		tid += blockDim.x * gridDim.x;
	}
}
