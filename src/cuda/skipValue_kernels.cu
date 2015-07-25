/*
 * skipValue_kernels.cu
 *
 *  Created: 2015-05-18, Modified: 2015-07-25
 *
 */

// Headers includes:
#include <cuda/kernels.cuh>
#include <cuda/Parallel_functions.h>

/** Prediction algorithm.
 * predicte the next edge that will be counted in the graph 
 * using compute intensive equation (has log inside it)
 */
__global__ void skipValue_kernel(float *Rands, 				/* in */ 
								 int size, 					/* in */ 
								 float skipping_prob,		/* in */ 
								 int *Skips)				/* out */
{
	int k;
	float logp;
    logp = log10f(1 - skipping_prob);
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
    while (tid < size) {
		k = max(0, (int)ceil( log10f(Rands[tid]) / logp ) );
		Skips[tid] = k;
		
		tid += blockDim.x * gridDim.x;
	}
}

//printf("S[%d] = %d\n", tid, S[tid]);
