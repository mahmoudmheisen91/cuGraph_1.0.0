#include <cuda/Parallel_functions.h> 

__global__ void addEdges_kernal(bool *content, float *S, int *V, int *B) {
	
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int v1, v2;
	
	while (tid < *B) {
	
		v1 = (int)S[tid] / *V;
		v2 = (int)S[tid] % *V;
		content[v1 * *V + v2] = 1;
		content[v2 * *V + v1] = 1;
		
		tid += blockDim.x * gridDim.x;
	}
	
}
