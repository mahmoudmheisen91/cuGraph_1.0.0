#include "main.hpp"

int main(int argc, char **argv) {

	// declare host and device variable:
	int *host_masterSeed, *host_itemsPerThread, *host_PRNG;
	int *dev_masterSeed, *dev_itemsPerThread, *dev_PRNG;
	
	// allocate host memory:
	host_masterSeed = new int[1];
	host_itemsPerThread = new int[1];
	host_PRNG = new int[N];
	
	// allocate device memory:
	cudaMalloc((void**)&dev_masterSeed, sizeof(int));
	cudaMalloc((void**)&dev_itemsPerThread, sizeof(int));
	cudaMalloc((void**)&dev_PRNG, N * sizeof(int));
	
	// fill 
	host_masterSeed[0] = 1;
	host_itemsPerThread[0] = 2;
	
	// copy host vars to device vars:
	cudaMemcpy(dev_masterSeed, host_masterSeed, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_itemsPerThread, host_itemsPerThread, sizeof(int), cudaMemcpyHostToDevice);
	
	// start kernal:
	random_number_generator_kernal<<<gridSize, blockSize>>>(dev_masterSeed, dev_itemsPerThread, dev_PRNG);
	
	// copy device vars to host vars:
	cudaMemcpy(host_PRNG, dev_PRNG, N * sizeof(int), cudaMemcpyDeviceToHost);
	
	// print output:
	for(int i = 0; i < N; i++) {
		cout << host_PRNG[i] << endl;
	}
	
	// free host memory:
	delete host_masterSeed;
	delete host_itemsPerThread;
	delete host_PRNG;
	
	// free device memory:
	cudaFree(dev_masterSeed);
	cudaFree(dev_itemsPerThread);
	cudaFree(dev_PRNG);
	
    return 0;
}

__global__ void random_number_generator_kernal(int *masterSeed, int *itemsPerThread, int *PRNG) {
	long int a = 16807;
	long int m = 2147483647;
	float rec = 1.0 / m;
	
	int seed = *masterSeed + threadIdx.x;
	
	int theta;
	long int temp;
	for (int i = 0; i < *itemsPerThread; i++) {
		temp = seed * a;
		theta = temp - m * floor(temp * rec);
		seed = theta;
		PRNG[i + *itemsPerThread*threadIdx.x] = theta;
	}
}








