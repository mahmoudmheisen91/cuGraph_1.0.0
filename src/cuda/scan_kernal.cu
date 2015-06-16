#include "mainPPS.hpp"

__global__ void scanGlobal(int *g_odata, int *g_idata, int n) {
	
}

__device__ void cuScanBlock(int *g_odata, int *g_idata, int n) {
	extern __shared__ float temp[];
	
	int thid = threadIdx.x;
	int offset = 1;
	
	temp[thid] = g_idata[thid];
	
	// build sum in place up the tree:
	for (int d = n>>1; d > 0; d >>= 1) { 
		__syncthreads();
		if (thid < d) {
			int ai = offset*(2*thid+1)-1;  
			int bi = offset*(2*thid+2)-1;  
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	
	// clear the last element:
	if (thid==0)
		temp[n - 1] = 0; 
	
	// traverse down tree & build scan:
	for (int d = 1; d < n; d *= 2) { 
		offset >>= 1;
		__syncthreads();
		if (thid < d) {
			int ai = offset*(2*thid+1)-1;  
			int bi = offset*(2*thid+2)-1;  
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();
	g_odata[thid] = temp[thid];
} 



