#include "main.hpp"

int main(int argc, char **argv) {

	// declare host and device variable:
	int *host_masterSeed, *host_itemsPerThread;
	float *host_PRNG;
	int *dev_masterSeed, *dev_itemsPerThread;
	float *PRNG, *dev_PRNG;
	
	// allocate host memory:
	int size = pow(2, 27);
	int a = size / (B*N);
	host_masterSeed = new int[1];
	host_itemsPerThread = new int[1];
	host_PRNG = new float[size];
	PRNG = new float[size];
	
	// allocate device memory:
	cudaMalloc((void**)&dev_masterSeed, sizeof(int));
	cudaMalloc((void**)&dev_itemsPerThread, sizeof(int));
	cudaMalloc((void**)&dev_PRNG, size * sizeof(float));
	
	// fill 
	host_masterSeed[0] = time(0);
	host_itemsPerThread[0] = a;
	
	// host kernal:
	//random_number_generator_host(PRNG);
	
	// copy host vars to device vars:
	cudaMemcpy(dev_masterSeed, host_masterSeed, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_itemsPerThread, host_itemsPerThread, sizeof(int), cudaMemcpyHostToDevice);
	
	float time1, time2;
	cudaEvent_t start, stop; 

	// start Device kernal:
	cudaEventCreate(&start); 
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0); 
	random_number_generator_kernal<<<gridSize, blockSize>>>(dev_masterSeed, dev_itemsPerThread, dev_PRNG);
	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop);  
	cudaEventElapsedTime(&time1, start, stop); 
	cudaEventDestroy(start); 
	cudaEventDestroy(stop);
	
	// start Host Kernal:
	cudaEventCreate(&start); 
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0); 
	random_number_generator_host(host_masterSeed, &size, PRNG);
	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop);  
	cudaEventElapsedTime(&time2, start, stop); 
	cudaEventDestroy(start); 
	cudaEventDestroy(stop);
	
	// print time:
	cout << time1 << " " << time2 << endl;
	
	// copy device vars to host vars:
	cudaMemcpy(host_PRNG, dev_PRNG, size * sizeof(float), cudaMemcpyDeviceToHost);
	
	// print output:
	/*
	int num = 10;
	int result[num];
	fill(result, result+num, 0);
	
	for(int i = 0; i < size; i++) {
		result[(int)(host_PRNG[i]*num)]++;
	}
	
	cout << "avg = " << size / num << endl;
	for(int i = 0; i < num; i++) {
		cout << result[i] << " ";
	}
	
	fill(result, result+num, 0);
	
	for(int i = 0; i < size; i++) {
		result[(int)(PRNG[i]*num)]++;
	}
	
	cout <<endl;
	for(int i = 0; i < num; i++) {
		cout << result[i] << " ";
	}*/
	
	// free host memory:
	delete host_masterSeed;
	delete host_itemsPerThread;
	delete host_PRNG;
	delete PRNG;
	
	// free device memory:
	cudaFree(dev_masterSeed);
	cudaFree(dev_itemsPerThread);
	cudaFree(dev_PRNG);
	
    return 0;
}

__global__ void random_number_generator_kernal(int *masterSeed, int *itemsPerThread, float *PRNG) {
	long int a = 16807;                      // same as apple c++ imp
	long int m = 2147483647;                 // 2^31 − 1
	float rec  = 1.0 / m;
	
	long int seed = *masterSeed + threadIdx.x + blockIdx.x;
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	long int theta;
	long int temp;
	int to = *itemsPerThread;
	for (int i = 0; i < to; i++) {
		temp = seed * a;                       // seed = Xn , c = 0
		theta = temp - m * floor(temp * rec);  // is the same as (temp mod m) ((Xn * a) mod m)
		seed = theta;
		PRNG[i + to * tid] = (float)theta/m;
	}
}


__host__ void random_number_generator_host(int *masterSeed, int *itemsPerThread, float *PRNG) {
	long int a = 16807;
	long int m = 2147483647;
	float rec = 1.0 / m;
	
	int threadId = 0;
	long int seed = (*masterSeed) + threadId;

	long int theta;
	long int temp;
	for (int i = 0; i < *itemsPerThread; i++) {
		temp = seed * a;
		theta = temp - m * floor(temp * rec);
		seed = theta;
		PRNG[i] = (float)theta/m;
	}
}





