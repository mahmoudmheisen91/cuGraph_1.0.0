#include "main.cuh"

__host__ void random_number_generator_host(int *masterSeed, int *itemsPerThread, float *PRNG) {
	long int a = 16807;
	long int m = 2147483647;
	float rec = 1.0 / m;
	
	int tid = 0;
	long int seed = (*masterSeed) + tid;

	long int theta;
	long int temp;
	int to = *itemsPerThread;
	for (int i = 0; i < to; i++) {
		temp = seed * a;
		theta = temp - m * floor(temp * rec);
		seed = theta;
		PRNG[i] = (float)theta/m;
	}
}

