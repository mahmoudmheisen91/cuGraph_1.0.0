#include "main.hpp"

void sample_run(int size, int num) {
	// declare host and device variable:
	int *host_masterSeed, *host_size, *dev_masterSeed, *dev_size;
	float *host_PRNG, *dev_PRNG, time1;
	cudaEvent_t start, stop; 
	dim3 grid(gridSize-1);
	dim3 block(blockSize);
	
	// allocate host memory:
	host_masterSeed = new int[1];
	host_size = new int[1];
	host_PRNG = new float[size];
	
	// allocate device memory:
	cudaMalloc((void**)&dev_masterSeed, sizeof(int));
	cudaMalloc((void**)&dev_size, sizeof(int));
	cudaMalloc((void**)&dev_PRNG, size * sizeof(float));
	
	// fill 
	host_masterSeed[0] = time(0) - 1000000000; // between 1 and m-1
	host_size[0] = size;
	
	// copy host vars to device vars:
	cudaMemcpy(dev_masterSeed, host_masterSeed, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_size, host_size, sizeof(int), cudaMemcpyHostToDevice);

	// start Device kernal:
	cudaEventCreate(&start); 
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0); 
	random_number_generator_kernal<<<grid, block>>>(dev_masterSeed, dev_size, dev_PRNG);
	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop);  
	cudaEventElapsedTime(&time1, start, stop); 
	cudaEventDestroy(start); 
	cudaEventDestroy(stop);
	
	// print time:
	cout << endl << "Kernal time = " << time1 << " ms, size = " << size << endl;
	
	// copy device vars to host vars:
	cudaMemcpy(host_PRNG, dev_PRNG, size * sizeof(float), cudaMemcpyDeviceToHost);
	
	float mn,mx;

	mn=host_PRNG[0];
	mx=host_PRNG[0];
	for(int i=1;i<size;i++)
	{
		if(mn>host_PRNG[i])
		{
			mn=host_PRNG[i];
		}
		else if(mx<host_PRNG[i])
		{
			mx = host_PRNG[i];
		}
	}
//cout<<"ti: "<< time(0) << endl;
	cout<<"Maximum number is: "<< mx << endl;
	cout<<"Minimum number is: "<< mn << endl;

	// print output:
	int result[num];
	fill(result, result+num, 0);
	int avg = size / num;
	for(int i = 0; i < size; i++) {
		//cout<< host_PRNG[i] << " ";
		result[(int)(host_PRNG[i]*num)]++;
	}
	
	for(int i = 0; i < 200; i++) {
		//cout<< host_PRNG[i] << " ";
	}
	cout << endl;
	for(int i = 0; i < num; i++) {
		cout << (100.0*result[i]/avg)/num << " ";
	}
	cout << endl << endl;
	
	// free host memory:
	delete host_masterSeed;
	delete host_size;
	delete host_PRNG;
	
	// free device memory:
	cudaFree(dev_masterSeed);
	cudaFree(dev_size);
	cudaFree(dev_PRNG);
}
