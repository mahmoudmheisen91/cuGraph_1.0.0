#include <cuda/Parallel_functions.h> 

__global__ void random_number_generator_kernal(int *masterSeed, int *size, float *PRNG);

void parallel_generateRandomNumber(float *R, int B, int seed) {

    // declare host and device variable:
	int *host_masterSeed, *host_size, *dev_masterSeed, *dev_size;
	float *dev_PRNG;
	
	dim3 grid(1);
	dim3 block(pow(2, 10));

	// allocate host memory:
	host_masterSeed = new int[1];
	host_size = new int[1];
	
	// allocate device memory:
	cudaMalloc((void**)&dev_masterSeed, sizeof(int));
	cudaMalloc((void**)&dev_size, sizeof(int));
	cudaMalloc((void**)&dev_PRNG, B * sizeof(float));
	
	// fill 
	host_masterSeed[0] = seed; // between 1 and m-1
	host_size[0] = B;
	
	// copy host vars to device vars:
	cudaMemcpy(dev_masterSeed, host_masterSeed, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_size, host_size, sizeof(int), cudaMemcpyHostToDevice);

	// start Device kernal:
	random_number_generator_kernal<<<grid, block>>>(dev_masterSeed, dev_size, dev_PRNG);
	
	// copy device vars to host vars:
	cudaMemcpy(R, dev_PRNG, B * sizeof(float), cudaMemcpyDeviceToHost);
	
	// free host memory:
	delete host_masterSeed;
	delete host_size;
	
	// free device memory:
	cudaFree(dev_masterSeed);
	cudaFree(dev_size);
	cudaFree(dev_PRNG);
}
