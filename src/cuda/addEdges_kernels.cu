/*
 * addEdges_kernels.cu
 *
 *  Created: 2015-05-18, Modified: 2015-07-25
 *
 */

// Headers includes:
#include <cuda/kernels.cuh>
#include <cuda/Parallel_functions.h>

/** Add Edges to the graph.
 * directed graph with self loops
 */
__global__ void addEdges_kernel(int *Skips, 		/* in */
								int skips_size,		/* in */
								int vertex_num,		/* in */
								bool *content,		/* out */
								int *L)				/* out */

{	
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int v1, v2;
    int Stid;
    int max_size = vertex_num * vertex_num;
	
	*L = S[skips_size-1];
	
    while (tid < skips_size) {
    	Stid = Skips[tid];
        v1 = Stid / vertex_num;
        v2 = Stid % vertex_num;
        content[(v1 * vertex_num + v2) % max_size] = true;
		
		tid += blockDim.x * gridDim.x;
	}
}

/** Update S array.
 * cancatate S from previous loop with current loop.
 */
__global__ void update_cancatate_kernel(int *Skips, 			/* in\out */ 
								 		int size,				/* in */ 
								 		int cancatate_val)	 	/* in */ 
{								 		
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
    while (tid < size) {
    	Skips[tid] = Skips[tid] + cancatate_val;
		
		tid += blockDim.x * gridDim.x;
	}
}

//printf("S[%d] = %d\n", tid, S[tid]);


