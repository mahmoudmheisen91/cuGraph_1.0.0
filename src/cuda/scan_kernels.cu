/*
 * scan_kernels.cu
 *
 *  Created: 2015-06-18, Modified: 2015-07-26
 *
 */

// Headers includes:
#include <cuda/kernels.cuh>
#include <cuda/Cuda_Prototypes_Macros.h>

/** Single warp scan algorithm.
 * device function that scan a single warp of threads
 */
__device__ int single_warp_scan(int *data, /* in\out */ 
								int idx)  /* in */ 
{
	const unsigned int tid = threadIdx.x;					 // thread id in a block: (0 to 1023)
	const unsigned int lane = tid & 31; 					 // index of thread in warp (0..31):
	
	// Unroll the loop:
	if ( lane >= 1)  
		data[idx] = data[idx] + data[idx - 1];
		
	if ( lane >= 2)  
		data[idx] = data[idx] + data[idx - 2];
		
	if ( lane >= 4)  
		data[idx] = data[idx] + data[idx - 4];
		
	if ( lane >= 8)  
		data[idx] = data[idx] + data[idx - 8];
		
	if ( lane >= 16) 
		data[idx] = data[idx] + data[idx - 16];
	
	return data[idx];
}

/** Single block scan algorithm.
 * device function that scan a single block of threads
 * internally depend on single_warp_scan
 */								
__device__ int single_block_scan(int *data, /* in\out */ 
								 int idx)   /* in */ 
{	
	const unsigned int tid = threadIdx.x;  	// thread id in a block: (0 to 1023)
	const unsigned int bid = blockIdx.x;  	// block id in a grid: (0 to 1023)
	const unsigned int bdim = blockDim.x;  	// block size: 1024
	const unsigned int lane = tid & 31;  	// thread id in a warp: (0 to 31)
	const unsigned int warpid = tid >> 5; 	// warp id in a block: (0 to 31)
	
	// Step 1: Single warp scan:
	int val = single_warp_scan(data, idx);
	__syncthreads();
		
	// Step 2: Collect partial results per warp:
	if( lane == 31 ) { 
		data[warpid + (bid*bdim) ] = data[idx]; // last thread in each warp
	}
	__syncthreads();
	
	// Step 3: Scan partail results:
	if( warpid == 0) {
		single_warp_scan(data, idx);
	}
	__syncthreads();

	// Step 4: Accumulate results from Steps 1 and 3:
	if (warpid > 0) {
		val = data[(warpid - 1) + (bid*bdim)] + val;
	}
	__syncthreads();
	
	// Step 5: Write and return the final result:
	data[idx] = val;
	__syncthreads();
	
	return val ;
}

/** First kernel of global scan algorithm.
 * scan data array as blocks
 * store partail results in block_results
 */										
__global__ void global_scan_kernel_1(int *data, 			/* in\out */ 
									 int *block_results)	/* out */ 
{
	const unsigned int tid = threadIdx.x;				// thread id in a block: (0 to 1023)
	const unsigned int bid = blockIdx.x;				// block id in a grid: (0 to 1023)
	const unsigned int gid = tid + bid * blockDim.x;  	// global id of the thread
	
	// step 1: block scan:
	int val = single_block_scan(data, gid);
	__syncthreads();
	
	// step 2: store partial result from each block:
	if (tid == 1023) {
		block_results[bid] = data[gid];
	}
	__syncthreads();
}

/** Second kernel of global scan algorithm.
 * scan block_results array
 */									
__global__ void global_scan_kernel_2(int *block_results)	/* in\out */ 
{
	const unsigned int tid = threadIdx.x;				// thread id in a block: (0 to 1023)
	
	// step 3: block scan of block_results:
	single_block_scan(block_results, tid);
}

/** Third kernel of global scan algorithm.
 * accumalte data from scanned block_results
 * store partail results in block_results
 */	
__global__ void global_scan_kernel_3(int *data, 			/* in\out */ 
									 int *block_results)	/* in */ 
{
	const unsigned int tid = threadIdx.x;				// thread id in a block: (0 to 1023)
	const unsigned int bid = blockIdx.x;				// block id in a grid: (0 to 1023)
	const unsigned int gid = tid + bid * blockDim.x;  	// global id of the thread
	
	// Step 4: Each thread of block i adds element i from Step 3 to its output element from Step 1:
	if (bid > 0) {
		int val = block_results[bid - 1];
		data[gid] += val;
	}
}






