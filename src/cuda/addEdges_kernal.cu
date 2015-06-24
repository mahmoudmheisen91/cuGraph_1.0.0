#include <cuda/Parallel_functions.h> 

__global__ void addEdges_kernal(bool *content, float *S, int V, int B, int *L, int last) {
	
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int v1, v2;
	
    while (tid < B) {
        S[tid] += last;
        v1 = (int)S[tid] / V;
        v2 = (int)S[tid] % V;
        content[v1 * V + v2] = 1;
        content[v2 * V + v1] = 1;
		
        if(tid == B-1) L[0] = S[tid];
		
		tid += blockDim.x * gridDim.x;
	}	
}

__global__ void addEdges_kernel2(bool *content, float *SC, int V, int B) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int v1, v2;

    while (tid < B) {

        if(SC[tid] > 0) {
            v1 = (int)SC[tid] / V;
            v2 = (int)SC[tid] % V;
            content[v1 * V + v2] = 1;
            content[v2 * V + v1] = 1;
        }

        tid += blockDim.x * gridDim.x;
    }
}

