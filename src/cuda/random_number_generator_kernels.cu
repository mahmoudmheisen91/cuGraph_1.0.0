/*
 * random_number_generator_kernels.cu
 *
 *  Created: 2015-05-18, Modified: 2015-07-25
 *
 */

// Headers includes:
#include <cuda/kernels.cuh>
#include <cuda/Parallel_functions.h>

/** Parallel random number generator.
 * with a master seed and local seed per thread, 
 * seed = masterseed + thread_global_id
 */
__global__ void random_number_generator_kernel(int masterSeed, /* in */ 
											   int size, 	   /* in */
											   float *PRNG)   /* out */
{
	long int a = 16807;                      	// same as apple c++ imp
	long int m = 2147483647;                 	// 2^31 − 1
	float rec  = 1.0 / m;
	long int theta, temp;
	
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
    long int seed = masterSeed + tid;         	// every thread has diffrent seed
	
    while (tid < size) {
		temp = seed * a;                       	// seed = Xn , c = 0
		theta = temp - m * floor(temp * rec);  	// is the same as (temp mod m) ((Xn * a) mod m)
		seed = theta;
		PRNG[tid] = (float)theta/m;			   	// between 1/m - 1
		
		tid += blockDim.x * gridDim.x;
	}
}

//printf("R[%d] = %.2f\n", tid, PRNG[tid]);



