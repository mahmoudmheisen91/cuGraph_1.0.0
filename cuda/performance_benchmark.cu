#include "main.cuh"

__host__ void performance_benchmark(int from, int to) {
	int *host_masterSeed, *host_itemsPerThread, *dev_masterSeed, *dev_itemsPerThread, a, size;
	float time1, time2, *host_PRNG, *dev_PRNG, *PRNG;
	cudaEvent_t start, stop; 
	
	host_masterSeed = new int[1];
	host_itemsPerThread = new int[1];
	cudaMalloc((void**)&dev_masterSeed, sizeof(int));
	cudaMalloc((void**)&dev_itemsPerThread, sizeof(int));
		
	cudaMemcpy(dev_masterSeed, host_masterSeed, sizeof(int), cudaMemcpyHostToDevice);
	
	for(int i = from; i <= to; i++) {
		
		size = pow(2, i);
		PRNG = new float[size];
		//cudaMallocHost ((void **) &host_PRNG, size);
		host_PRNG = new float[size];
		cudaMalloc((void**)&dev_PRNG, size * sizeof(float));
		
		a = size / ((B-1)*N);
		host_masterSeed[0] = time(0);
		host_itemsPerThread[0] = a;
		cudaMemcpy(dev_itemsPerThread, host_itemsPerThread, sizeof(int), cudaMemcpyHostToDevice);

		cudaEventCreate(&start); 
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0); 
		random_number_generator_kernal<<<grid, block>>>(dev_masterSeed, dev_itemsPerThread, dev_PRNG);
		cudaEventRecord(stop, 0); 
		cudaEventSynchronize(stop);  
		cudaEventElapsedTime(&time1, start, stop); 
		cudaEventDestroy(start); 
		cudaEventDestroy(stop);
		
		cudaMemcpy(host_PRNG, dev_PRNG, size * sizeof(float), cudaMemcpyDeviceToHost);
		
		cudaEventCreate(&start); 
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0); 
		random_number_generator_host(host_masterSeed, &size, PRNG);
		cudaEventRecord(stop, 0); 
		cudaEventSynchronize(stop);  
		cudaEventElapsedTime(&time2, start, stop); 
		cudaEventDestroy(start); 
		cudaEventDestroy(stop);
		
		delete host_PRNG;
		//cudaFree(host_PRNG);
		cudaFree(dev_PRNG);
		delete PRNG;
		
		cout << "Size = 2^" << i << " : "<< "Time1 = " << time1 << ", Time2 = " << time2;
		cout <<  ", increae: X" << (int)(time2/time1) << endl;
	}
		
	delete host_masterSeed;
	delete host_itemsPerThread;
	
	cudaFree(dev_masterSeed);
	cudaFree(dev_itemsPerThread);
	
}










