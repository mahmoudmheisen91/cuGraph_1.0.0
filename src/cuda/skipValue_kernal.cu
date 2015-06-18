#include <cuda/Parallel_functions.h> 

__global__ void skipValue_kernal(int *S, float *R, int *B, float *p) {

	int k;
	float logp;
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	while (tid < *B) {
		logp = log10f(R[tid]) / log10f(1- *p);
		
		//k = max(0, (int)ceil(logp) - 1);
		k = max(0, (int)ceil(logp) - 1);
		
		if (k < 0)
			k = 0;
			
		S[tid] = k;
		tid += blockDim.x * gridDim.x;
	}
	
}
