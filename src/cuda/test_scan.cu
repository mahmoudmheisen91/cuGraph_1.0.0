#include "mainPPS.hpp"

void test_scan(int size, int num) {
	// declare host and device variable:
	int *host_in, *host_out, *dev_in, *dev_out, n = size;
	float time1;
	cudaEvent_t start, stop; 
	
	dim3 grid(1);
	dim3 block(n);

	// allocate host memory:
	host_in = new int[n];
	host_out = new int[n];
	
	// allocate device memory:
	cudaMalloc((void**)&dev_in, n * sizeof(int));
	cudaMalloc((void**)&dev_out, n * sizeof(int));
	
	// fill 
	for(int i = 0; i < n; i++) {
		host_in[i] = i+1;
	}
	
	// copy host vars to device vars:
	cudaMemcpy(dev_in, host_in, n*sizeof(int), cudaMemcpyHostToDevice);
int shared = n*sizeof(int);
	// start Device kernal:
	cudaEventCreate(&start); 
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0); 
	cuScan<<<grid, block>>>(dev_out, dev_in, n);
	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop);  
	cudaEventElapsedTime(&time1, start, stop); 
	cudaEventDestroy(start); 
	cudaEventDestroy(stop);
	
	// print time:
	cout << endl << "Kernal time = " << time1 << " ms, size = " << n << endl;
	
	// copy device vars to host vars:
	cudaMemcpy(host_out, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);
	
	for(int i = 0; i < n; i++) {
		cout << host_out[i] << " ";
	}
	
	cout << endl;
	// free host memory:
	delete host_in;
	delete host_out;
	
	// free device memory:
	cudaFree(dev_in);
	cudaFree(dev_out);
}
