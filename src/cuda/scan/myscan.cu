#include <iostream>
#include <cuda.h>
#include <cstdio>

using namespace std;

__device__ int single_warp_scan(int *data_in, int idx);
__device__ int single_block_scan(int *data_in, int idx);
__global__ void global_scan_kernel_1(int *data_in, int *block_results);

int main() {

	// params:
	int size = 2048;
	
	// allocate host:
	int *data_in_host = NULL;
	data_in_host = new int[size];
	
	int *data_out_host = NULL;
	data_out_host = new int[size];
	
	// allocate device:
	int *data_in_device = NULL;
	cudaMalloc((void**) &data_in_device, size * sizeof(int));
	
	int *block_results = NULL;
	cudaMalloc((void**) &block_results, 1024 * sizeof(int));
	
	// fill host:
	for(int i = 0; i < size; i++) {
		data_in_host[i] = 1;
	}
	
	// copy host to device:
	cudaMemcpy(data_in_device, data_in_host, size * sizeof(int), cudaMemcpyHostToDevice);
	
	// kernel:
	global_scan_kernel_1 <<<2, 1024>>> (data_in_device, block_results);
	//single_block_scan <<<1, pow(2, 10)>>> (data_in_device, data_out_device, size, items_per_thread);
	
	// copy device to host:
	cudaMemcpy(data_out_host, data_in_device, size * sizeof(int), cudaMemcpyDeviceToHost);
	
	// print:
	for(int i = 0; i < size; i++) {
		cout << data_out_host[i] << " ";
	}
	cout << endl;
	
	// free:
	delete data_in_host;
	delete data_out_host;
	cudaFree(data_in_device);
	
	// end:
	return 0;
}

__device__ int single_warp_scan(int *data_in, int idx) {

	//int idx = threadIdx.x;
	const unsigned int lane = idx & 31; // index of thread in warp (0..31)
	
	if ( lane >= 1)  data_in[idx] = data_in[idx] + data_in[idx - 1];
	if ( lane >= 2)  data_in[idx] = data_in[idx] + data_in[idx - 2];
	if ( lane >= 4)  data_in[idx] = data_in[idx] + data_in[idx - 4];
	if ( lane >= 8)  data_in[idx] = data_in[idx] + data_in[idx - 8];
	if ( lane >= 16) data_in[idx] = data_in[idx] + data_in[idx - 16];
	
	//printf("%d ", lane);
	return data_in[idx];
}

__device__ int single_block_scan(int *data_in, int idx) {

	//int idx = threadIdx.x;
	const unsigned int lane = idx & 31;
	const unsigned int warpid = idx >> 5;
	
	// Step 1: Intra - warp scan in each warp
	int val = single_warp_scan(data_in, idx);
	__syncthreads ();
	
	// Step 2: Collect per - warp partial results
	if( lane ==31 ) data_in[warpid] = data_in[idx];
	__syncthreads ();
	
	// Step 3: Use 1 st warp to scan per - warp results
	if( warpid ==0 ) single_warp_scan(data_in, idx);
	__syncthreads ();
	
	// Step 4: Accumulate results from Steps 1 and 3
	if (warpid > 0) val = data_in[warpid - 1] + val;
	__syncthreads ();
	
	// Step 5: Write and return the final result
	data_in[idx] = val;
	__syncthreads ();
	
	return val ;
}

__global__ void global_scan_kernel_1(int *data_in,int *block_results) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int gid = tid + bid * blockDim.x;
	
	// step 1: block scan:
	int val = single_block_scan(data_in, gid);
	__syncthreads ();
	
	// step 2: store partial result from each block:
	if (tid == 1023) {
		block_results[bid] = data_in[gid];
		//printf("%d ", block_results[bid]);
	}
	
}








