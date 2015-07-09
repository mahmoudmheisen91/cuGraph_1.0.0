#include <iostream>
#include <cuda.h>
#include <cstdio>

using namespace std;

__device__ int single_warp_scan(int *data_in, int idx);
__device__ int single_block_scan(int *data_in, int idx);
__global__ void global_scan_kernel_1(int *data_in, int *block_results);
__global__ void global_scan_kernel_2(int *block_results);
__global__ void global_scan_kernel_3(int *data_in, int *block_results);

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

__device__ int single_warp_scan(int *data_in, int idx) {

	int tid = threadIdx.x;
	const unsigned int lane = tid & 31; // index of thread in warp (0..31)
	
	if ( lane >= 1)  data_in[idx] = data_in[idx] + data_in[idx - 1];
	if ( lane >= 2)  data_in[idx] = data_in[idx] + data_in[idx - 2];
	if ( lane >= 4)  data_in[idx] = data_in[idx] + data_in[idx - 4];
	if ( lane >= 8)  data_in[idx] = data_in[idx] + data_in[idx - 8];
	if ( lane >= 16) data_in[idx] = data_in[idx] + data_in[idx - 16];
	
	//printf("%d ", lane);
	return data_in[idx];
}

__device__ int single_block_scan(int *data_in, int idx) {

	int tid = threadIdx.x;
	const unsigned int lane = tid & 31;
	const unsigned int warpid = tid >> 5;
	
	// Step 1: Intra - warp scan in each warp
	int val = single_warp_scan(data_in, idx);
	__syncthreads ();
		
	// Step 2: Collect per - warp partial results
	if( lane == 31 ) { 
		data_in[warpid+blockIdx.x*1024] = data_in[idx]; // last thread in each warp
		//printf("%d ", warpid+blockIdx.x*1024);
	}
	__syncthreads ();
	
	// Step 3: Use 1 st warp to scan per - warp results
	if( warpid == 0) {
		single_warp_scan(data_in, idx);
		//printf("warpid: %d, data_in[%d]: %d\n", warpid, idx, data_in[idx]);
	}
	__syncthreads ();

	// Step 4: Accumulate results from Steps 1 and 3
	if (warpid > 0) val = data_in[(warpid - 1)+blockIdx.x*1024] + val;
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

__global__ void global_scan_kernel_2(int *block_results) {

	int tid = threadIdx.x;
	
	// step 3: block scan of block_results:
	int val = single_block_scan(block_results, tid);
	//__syncthreads ();
	
	//if (tid == 0) {
	//	for(int i = 0; i <1024; i++)
	//		printf("%d ", block_results[i]);
	//}
}

__global__ void global_scan_kernel_3(int *data_in, int *block_results) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int gid = tid + bid * blockDim.x;
	
	// Step 4: Each thread of block i adds element i from Step 3 to its output element from Step 1:
	if (bid > 0) {
		int val = block_results[bid - 1];
		data_in[gid] += val;
	}
	
	__syncthreads ();
}











