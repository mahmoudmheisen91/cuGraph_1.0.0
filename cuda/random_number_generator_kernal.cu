#include "main.hpp"

__global__ void random_number_generator_kernal(int *masterSeed, int *size, float *PRNG) {
	long int a = 16807;                      // same as apple c++ imp
	long int m = 2147483647;                 // 2^31 − 1
	float rec  = 1.0 / m;
	
	//int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = (blockIdx.y * gridDim.x + blockIdx.x)*(blockDim.y * blockDim.x) + threadIdx.y * blockDim.x + threadIdx.x; 
	
	//int tid = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x)*blockDim.x); 
	long int seed = *masterSeed + tid;
	
	long int theta, temp;
	if (tid < *size) {
		temp = seed * a;                       // seed = Xn , c = 0
		theta = temp - m * floor(temp * rec);  // is the same as (temp mod m) ((Xn * a) mod m)
		//seed = theta;
		PRNG[tid] = (float)theta/m;
	}
}
