#include "main.cuh"

__host__ void  grid_block_test_case(int numB, int numT) {
	int *host_masterSeed, *host_itemsPerThread, *dev_masterSeed, *dev_itemsPerThread, b, n, a;
	float time1, *host_PRNG, *dev_PRNG;
	cudaEvent_t start, stop; 
	
	int size = pow(2, 27);
	
	host_masterSeed = new int[1];
	host_masterSeed[0] = time(0);
	host_itemsPerThread = new int[1];
	host_PRNG = new float[size];
	//cudaMallocHost ((void **) &host_PRNG, size);
	cudaMalloc((void**)&dev_masterSeed, sizeof(int));
	cudaMalloc((void**)&dev_itemsPerThread, sizeof(int));
	cudaMalloc((void**)&dev_PRNG, size * sizeof(float));
		
	cudaMemcpy(dev_masterSeed, host_masterSeed, sizeof(int), cudaMemcpyHostToDevice);
	
	for(int i = 1; i <= numB; i++) {
		for(int j = 1; j <= numT; j++) {
			b = pow(2, i);
			n = pow(2, j);
			a = size / ((b-1)*n);
			host_itemsPerThread[0] = a;
			cudaMemcpy(dev_itemsPerThread, host_itemsPerThread, sizeof(int), cudaMemcpyHostToDevice);
			
			cudaEventCreate(&start); 
			cudaEventCreate(&stop);
			cudaEventRecord(start, 0); 
			random_number_generator_kernal<<<b-1, n>>>(dev_masterSeed, dev_itemsPerThread, dev_PRNG);
			cudaEventRecord(stop, 0); 
			cudaEventSynchronize(stop);  
			cudaEventElapsedTime(&time1, start, stop); 
			cudaEventDestroy(start); 
			cudaEventDestroy(stop);
			cout << "Time = " << time1 << " : " << "numB = " << i << ", numT = " << j << endl;
		}
	}
	
	cudaMemcpy(host_PRNG, dev_PRNG, size * sizeof(float), cudaMemcpyDeviceToHost);
	
	delete host_masterSeed;
	delete host_itemsPerThread;
	delete host_PRNG;
	//cudaFree(host_PRNG);
	cudaFree(dev_masterSeed);
	cudaFree(dev_itemsPerThread);
	cudaFree(dev_PRNG);
}










