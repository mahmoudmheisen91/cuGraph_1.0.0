/*
 * stream_compaction_kernel.cu
 *
 *  Created: 2015-07-26, Modified: 
 *
 */

// Headers includes:
#include <cuda/kernels.cuh>
#include <cuda/Cuda_Prototypes_Macros.h>

/** Stream Compcation Algorithm.
 * directed graph with self loops
 */
__global__ void stream_compaction_kernel(int *T, 				/* in */
										 int *S, 				/* in */
										 int *predicate_list, 	/* in */
										 int size,				/* in */
										 int *SC)				/* out */
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while(tid < size) {

        if(predicate_list[tid] == 1) {
            int j = S[tid];
            SC[j] = T[tid];
        }

        tid += blockDim.x * gridDim.x;
    }
}










