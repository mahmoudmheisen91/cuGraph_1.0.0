/*
 * kernels.cuh
 *
 *  Created: 2015-07-24, Modified: 2015-07-25
 *
 */
 
#ifndef KERNELS_CUH_
#define KERNELS_CUH_

// Device flag to use blocking synchronization:
#define cudaDeviceScheduleBlockingSync   0x04

/** Parallel random number generator.
 * with a master seed and local seed per thread, 
 * seed = masterseed + thread_global_id
 */
__global__ void random_number_generator_kernel(int masterSeed, /* in */ 
											   int size, 	   /* in */
											   float *PRNG);   /* out */

/** Prediction algorithm.
 * predicte the next edge that will be counted in the graph 
 * using compute intensive equation (has log inside it)
 */
__global__ void skipValue_kernel(float *Rands, 				/* in */ 
								 int size, 					/* in */ 
								 float skipping_prob		/* in */ 
								 int *Skips);				/* out */

/** Single warp scan algorithm.
 * device function that scan a single warp of threads
 */
__device__ int single_warp_scan(int *data, /* in\out */ 
								int idx);  /* in */ 

/** Single block scan algorithm.
 * device function that scan a single block of threads
 * internally depend on single_warp_scan
 */								
__device__ int single_block_scan(int *data, /* in\out */ 
								 int idx);  /* in */ 

/** First kernel of global scan algorithm.
 * scan data array as blocks
 * store partail results in block_results
 */										
__global__ void global_scan_kernel_1(int *data, 			/* in\out */ 
									 int *block_results);	/* out */ 
									
/** Second kernel of global scan algorithm.
 * scan block_results array
 */									
__global__ void global_scan_kernel_2(int *block_results);	/* in\out */ 

/** Third kernel of global scan algorithm.
 * accumalte data from scanned block_results
 * store partail results in block_results
 */	
__global__ void global_scan_kernel_3(int *data, 			/* in\out */ 
									 int *block_results);	/* in */ 

/** Update S array.
 * cancatate S from previous loop with current loop.
 */
__global__ void update_cancatate(int *Skips, 			/* in\out */ 
								 int size,				/* in */ 
								 int cancatate_val); 	/* in */ 
								 

/** Add Edges to the graph.
 * directed graph with self loops
 */
__global__ void addEdges_kernel(int *Skips, 		/* in */
								int skips_size,		/* in */
								int vertex_num,		/* in */
								bool *content,		/* out */
								int *L);			/* out */

// unfactored code:								 
__global__ void skipValuePre_kernal(float *S, float *R, int B, float p, int m, float *F);
__global__ void generatePredicateList_kernel(float *PL, int *T, float *R, int B, int i, float p);
__global__ void compact_kernel(int *T, float *S, float *PL, int *SC, int B);
__global__ void addEdges_kernel2(bool *content, float *SC, int V, int B);

#endif












