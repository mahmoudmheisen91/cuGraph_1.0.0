#include <iostream>
#include <cuda.h>
#include <cstdio>

#include "scan_kernels.cuh"

using namespace std;

int main() {

	// params:
	int size = 1024*1024;
	
	// allocate host:
	int *data_host = NULL;
	data_host = new int[size];
	
	// allocate device:
	int *data_device = NULL;
	cudaMalloc((void**) &data_device, size * sizeof(int));
	
	int *block_results = NULL;
	cudaMalloc((void**) &block_results, 1024 * sizeof(int));
	
	// fill host:
	for(int i = 0; i < size; i++) {
		data_host[i] = 1;
	}
	
	// copy host to device:
	cudaMemcpy(data_device, data_host, size * sizeof(int), cudaMemcpyHostToDevice);
	
	// kernel:
	
	float time;
    cudaEvent_t start, stop; 

    cudaEventCreate(&start); 
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0); 

	global_scan_kernel_1 <<<1024, 1024>>> (data_device, block_results);
	global_scan_kernel_2 <<<1, 1024>>> (block_results);
	global_scan_kernel_3 <<<1024, 1024>>> (data_device, block_results);
	
    cudaEventRecord(stop, 0); 
    cudaEventSynchronize(stop);  
    cudaEventElapsedTime(&time, start, stop); 
    cudaEventDestroy(start); 
    cudaEventDestroy(stop);
	
	// copy device to host:
	cudaMemcpy(data_host, data_device, size * sizeof(int), cudaMemcpyDeviceToHost);
	
	// print:
	cout << "time = " << time << endl;
	for(int i = size-1; i < size; i++) {
		cout << data_host[i] << " ";
	}
	cout << endl;
	
	// free:
	delete data_host;
	cudaFree(data_device);
	
	// end:
	return 0;
}












