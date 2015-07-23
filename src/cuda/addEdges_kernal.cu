#include <cuda/Parallel_functions.h> 

#include <cstdio>

__global__ void addEdges_kernal(bool *content, int *S, int V, int B, int *d_L) {
	
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int v1, v2;
    int Stid;
	
	//if (tid == 0)
	*d_L = S[B-1];
	//printf("d_L[0] = %d\n", d_L[0]);
	
    while (tid < B) {
    	Stid = S[tid];
        v1 = Stid / V;
        v2 = Stid % V;
        content[v1 * V + v2] = 1;
		
		tid += blockDim.x * gridDim.x;
	}
}

__global__ void modify_S(int *S, int L, int B) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
    while (tid < B) {
    	S[tid] = S[tid] + L;
		
		//printf("S[%d] = %d\n", tid, S[tid]);
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

