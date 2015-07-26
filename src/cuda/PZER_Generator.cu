/*
 * PZER_Generator.cu
 *
 *  Created: 2015-05-10, Modified: 2015-07-26
 *
 */

// Must be defined before any header to use GPU_TIMER:
// Must be defined in just one file of the project (to be fixed):
//#define TIMER

// Headers includes:
#include <cuda/kernels.cuh>
#include <cuda/Cuda_Prototypes_Macros.h>

// Parallel CUDA random graph generator 2, PZER:
void PZER_Generator(bool *content,   		/* in\out */ 
				   	float skipping_prob,	/* in */ 
				   	int lambda, 			/* in */ 
				   	int vertex_num, 		/* in */ 
				   	int edges_num)			/* in */ 
{
    // Const:
    const unsigned int B = GRID_SIZE * BLOCK_SIZE;
    const unsigned int seed = time(0) - 1000000000;
    
    // Declerations:
    bool *d_content; float *d_R;
    int *d_L, *d_S, *d_block_results, L;   

    // Allocations:
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_content, vertex_num * vertex_num * sizeof(bool)));
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_R, B * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_S, B * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_L, sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_block_results, GRID_SIZE * sizeof(int)));

    // Init d_content with false values using cudaMemset:
    CUDA_SAFE_CALL(cudaMemset(d_content, false, vertex_num * vertex_num * sizeof(bool)));

    // Main loop:
    L = 0;
    while(L < edges_num) {
        random_number_generator_kernel 	<<<8		 , BLOCK_SIZE>>> (seed, B, d_R);						// Generate B rand numbers
        skipValue_kernel				<<<GRID_SIZE , BLOCK_SIZE>>> (d_R, B, skipping_prob, d_S);			// predicte skiped edges
		global_scan_kernel_1 			<<<GRID_SIZE , BLOCK_SIZE>>> (d_S, d_block_results);				// scan k1
		global_scan_kernel_2 			<<<1		 , BLOCK_SIZE>>> (d_block_results);						// scan k2
		global_scan_kernel_3 			<<<GRID_SIZE , BLOCK_SIZE>>> (d_S, d_block_results);				// scan k3
        update_cancatate_kernel			<<<GRID_SIZE , BLOCK_SIZE>>> (d_S, B, L);							// concanate scaned values
        addEdges_kernel					<<<GRID_SIZE , BLOCK_SIZE>>> (d_S, B, vertex_num, d_content, d_L);	// add edge to the graph
		
		// Update the value of L:
        CUDA_SAFE_CALL(cudaMemcpy(&L, d_L, sizeof(int), cudaMemcpyDeviceToHost));	
    }

	// Copy content from Device To Host:
    CUDA_SAFE_CALL(cudaMemcpy(content, d_content, sizeof(bool) * vertex_num * vertex_num, cudaMemcpyDeviceToHost));

    // Free the device:
    CUDA_SAFE_CALL(cudaFree(d_S));
    CUDA_SAFE_CALL(cudaFree(d_R));
    CUDA_SAFE_CALL(cudaFree(d_L));
    CUDA_SAFE_CALL(cudaFree(d_content));
    CUDA_SAFE_CALL(cudaFree(d_block_results));
    CUDA_SAFE_CALL(cudaDeviceReset());
}





