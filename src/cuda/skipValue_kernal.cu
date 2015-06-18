#include <cuda/Parallel_functions.h> 

__global__ void skipValue_kernal(float *S, float *R, int *B, float *p) {

	int k;
	float theta, logp;
	logp = log10f(1 - *p);
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	while (tid < *B) {
		theta = log10f(R[tid]) / logp;
		k = max(0, (int)ceil(theta) - 1);
		
		S[tid] = k;
		tid += blockDim.x * gridDim.x;
	}
}
