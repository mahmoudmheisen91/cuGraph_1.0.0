/*
 * skipValue_kernels.cu
 *
 *  Created: 2015-05-18, Modified: 2015-07-25
 *
 */

// Headers includes:
#include <cuda/kernels.cuh>
#include <cuda/Cuda_Prototypes_Macros.h>

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

/** Prediction algorithm.
 * predicte the next edge that will be counted in the graph 
 * using cumulative distribution function
 */
__global__ void skipValuePre_kernel(float *Rands, 			/* in */ 
									int size, 				/* in */ 
									float skipping_prob, 	/* in */ 
									int m, 					/* in */ 
									float *cumulative_dist,	/* in */ 
									int *Skips)			/* in */ 
{
	int k;
	float logp;
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
		
    while (tid < size) {
		int j = 0;
		
		// loop insead log function, because log is costly:
        while (j <= m) {
			if(cumulative_dist[j] > Rands[tid]) {
				k = j;
				break;       // to break from while loop;
			}
			else
				j++;
		}
		
		// not always called, just in case if j exceeded m:
        if(j == m + 1) {
        	logp = log10f(1 - skipping_prob);
			k = max(0, (int)ceil( log10f(Rands[tid]) / logp ));
		}
		
		Skips[tid] = k;
		
		tid += blockDim.x * gridDim.x;
	}
}

//printf("S[%d] = %d\n", tid, S[tid]);










