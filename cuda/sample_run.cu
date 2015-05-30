#include "main.hpp"

void sample_run(int size, int num) {
	// declare host and device variable:
	int *host_masterSeed, *host_itemsPerThread, *dev_masterSeed, *dev_itemsPerThread;
	float *host_PRNG, *dev_PRNG, time1;
	cudaEvent_t start, stop; 
	dim3 grid(B-1, B-1, 1);
	dim3 block(N, 1, 1);
	
	// allocate host memory:
	int a = size / ((B-1)*(B-1)*N);
	host_masterSeed = new int[1];
	host_itemsPerThread = new int[1];
	host_PRNG = new float[size];
	
	// allocate device memory:
	cudaMalloc((void**)&dev_masterSeed, sizeof(int));
	cudaMalloc((void**)&dev_itemsPerThread, sizeof(int));
	cudaMalloc((void**)&dev_PRNG, size * sizeof(float));
	
	// fill 
	host_masterSeed[0] = time(0);
	host_itemsPerThread[0] = a;
	
	// copy host vars to device vars:
	cudaMemcpy(dev_masterSeed, host_masterSeed, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_itemsPerThread, host_itemsPerThread, sizeof(int), cudaMemcpyHostToDevice);

	// start Device kernal:
	cudaEventCreate(&start); 
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0); 
	random_number_generator_kernal<<<grid, block>>>(dev_masterSeed, dev_itemsPerThread, dev_PRNG);
	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop);  
	cudaEventElapsedTime(&time1, start, stop); 
	cudaEventDestroy(start); 
	cudaEventDestroy(stop);
	
	// print time:
	cout << endl << "Kernal time = " << time1 << " ms, size = " << size << endl;
	
	// copy device vars to host vars:
	cudaMemcpy(host_PRNG, dev_PRNG, size * sizeof(float), cudaMemcpyDeviceToHost);
	
	// print output:
	int result[num];
	fill(result, result+num, 0);
	int avg = size / num;
	for(int i = 0; i < size; i++) {
		result[(int)(host_PRNG[i]*num)]++;
	}
	
	for(int i = 0; i < num; i++) {
		cout << (100.0*result[i]/avg)/num << " ";
	}
	cout << endl << endl;
	
	// free host memory:
	delete host_masterSeed;
	delete host_itemsPerThread;
	delete host_PRNG;
	
	// free device memory:
	cudaFree(dev_masterSeed);
	cudaFree(dev_itemsPerThread);
	cudaFree(dev_PRNG);
}
