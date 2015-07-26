/*
 * addEdges_kernels.cu
 *
 *  Created: 2015-05-18, Modified: 2015-07-25
 *
 */

// Headers includes:
#include <cuda/kernels.cuh>
#include <cuda/Cuda_Prototypes_Macros.h>

/** Add Edges to the graph.
 * directed graph with self loops
 */
__global__ void addEdges_kernel(int *Skips, 		/* in */
								int skips_size,		/* in */
								int vertex_num,		/* in */
								bool *content,		/* out */
								int *L)				/* out */

{	
	int Stid;
    int v1, v2;
    int max_size = vertex_num * vertex_num;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(tid == 0) {
		L[0] = Skips[skips_size-1];
	}
	
    while (tid < skips_size) {
    	Stid = Skips[tid];
        v1 = Stid / vertex_num;
        v2 = Stid % vertex_num;
        content[(v1 * vertex_num + v2) % max_size] = true;
		
		tid += blockDim.x * gridDim.x;
	}
}

/** Add Edges to the graph.
 * directed graph with self loops
 */
__global__ void addEdges_kernel_2(int *predicate_list, 	/* in */
								  int skips_size, 		/* in */
								  int vertex_num, 		/* in */
								  bool *content)		/* out */
{
	int Stid;
    int v1, v2;
    int max_size = vertex_num * vertex_num;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < skips_size) {
		Stid = predicate_list[tid];
		
        if(Stid > 0) {
            v1 = Stid / vertex_num;
            v2 = Stid % vertex_num;
            content[(v1 * vertex_num + v2) % max_size] = true;
        }

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


