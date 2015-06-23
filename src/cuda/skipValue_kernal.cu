#include <cuda/Parallel_functions.h> 

__global__ void skipValue_kernal(float *S, float *R, int B, float p) {

	int k;
	float theta, logp;
    logp = log10f(1 - p);
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
    while (tid < B) {
		theta = log10f(R[tid]) / logp;
		k = max(0, (int)ceil(theta) );
		
		S[tid] = k;
		tid += blockDim.x * gridDim.x;
	}
}

__global__ void skipValuePre_kernal(float *S, float *R, int *B, float *p, int *m, float *F) {
	
	int k;
	float theta;
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
		
	while (tid < *B) {
		
		int j = 0;
		while (j <= *m) {
			if(F[j] > R[tid]) {
				k = j;
				break;       // to break from while loop;
			}
			else
				j++;
		}
		
		if(j == *m + 1) {
			theta = log10f(R[tid]) / log10f(1 - *p);
			k = max(0, (int)ceil(theta) );
		}
		
		S[tid] = k;
		tid += blockDim.x * gridDim.x;
	}
}
