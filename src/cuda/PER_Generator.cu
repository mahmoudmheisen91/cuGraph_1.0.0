/*
 * PER_Generator.cu
 *
 *  Created: 2015-07-26, Modified: 
 *
 */

// Must be defined before any header to use GPU_TIMER:
// Must be defined in just one file of the project (to be fixed):
//#define TIMER

// Headers includes:
#include <cuda/kernels.cuh>
#include <cuda/Cuda_Prototypes_Macros.h>

// Parallel CUDA random graph generator 1, PER:
void PER_Generator(bool *content, 			/* in\out */ 
				   float skipping_prob, 	/* in */ 
				   int vertex_num, 		 	/* in */ 
				   int edges_num)		 	/* in */ 
{
	// Const:
    const unsigned int B = GRID_SIZE * BLOCK_SIZE;
    const unsigned int seed = time(0) - 1000000000;
    
    // Declerations:
    bool *d_content; float *d_R;
    int *d_PL, *d_T, *d_SC;
    int iter = edges_num / B;

	// Allocations:
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_content, vertex_num * vertex_num * sizeof(bool)));
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_R, B * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_T, B * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_PL, B * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_SC, B * sizeof(int)));

	// Init d_content with false values using cudaMemset:
    CUDA_SAFE_CALL(cudaMemset(d_content, false, vertex_num * vertex_num * sizeof(bool)));

    // run kernals:
    for(int i = 0; i < iter; i++) {
        random_number_generator_kernel 	<<<8		 , BLOCK_SIZE>>> (seed, B, d_R);						// Generate B rand numbers
        generate_predicate_list_kernel	<<<GRID_SIZE , BLOCK_SIZE>>> (d_R, B, skipping_prob, i, d_PL, d_T); // Generate predicate list
        addEdges_kernel_2				<<<GRID_SIZE , BLOCK_SIZE>>> (d_PL, B, vertex_num, d_content);		// add edge to the graph
    }

	// Copy content from Device To Host:
    CUDA_SAFE_CALL(cudaMemcpy(content, d_content, sizeof(bool) * vertex_num * vertex_num, cudaMemcpyDeviceToHost));

	// Free the device:
    CUDA_SAFE_CALL(cudaFree(d_R));
    CUDA_SAFE_CALL(cudaFree(d_T));
    CUDA_SAFE_CALL(cudaFree(d_SC));
    CUDA_SAFE_CALL(cudaFree(d_PL));
    CUDA_SAFE_CALL(cudaFree(d_content));
    CUDA_SAFE_CALL(cudaDeviceReset());
}










