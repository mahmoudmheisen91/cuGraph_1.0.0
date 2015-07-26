/*
 * Parallel_functions.h
 *
 *  Created: 2015-05-01, Modified: 2015-07-26
 *
 */
 
#ifndef PARALLEL_FUNCTIONS_H_
#define PARALLEL_FUNCTIONS_H_

// Standard C libraries includes:
#include <cstdio>
#include <cstdlib>

// Standard C++ libraries includes:
#include <iostream>
#include <iomanip>
using namespace std;

// Include header file for CUDA Runtime API:
#include <cuda_runtime_api.h>

// Define the grid size:
#define GRID_SIZE 1024

// Define the block size:
#define BLOCK_SIZE 1024

// Macro to catch CUDA errors in CUDA runtime calls:
#define CUDA_SAFE_CALL(err) \
    if (cudaSuccess != err) { \
       	fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \
           	     __FILE__, __LINE__, cudaGetErrorString(err) ); \
    	exit(EXIT_FAILURE); \
    } \

// Decision for GPU_TIMER globals: 
// in the header file declare it with extern, in c files declare it without extern:
#ifdef TIMER
	cudaEvent_t start, stop; 
#else
	extern cudaEvent_t start, stop; 
#endif

// GPU_TIMER macro to start the timer:
#define GPU_TIMER_START(); \
    CUDA_SAFE_CALL(cudaEventCreate(&start));   \
    CUDA_SAFE_CALL(cudaEventCreate(&stop));    \
    CUDA_SAFE_CALL(cudaEventRecord(start, 0)); 

// GPU_TIMER macro to return the elapsed time:
#define GPU_TIMER_END(time); \
    CUDA_SAFE_CALL(cudaEventRecord(stop, 0)); \
	CUDA_SAFE_CALL(cudaEventSynchronize(stop)); \
	CUDA_SAFE_CALL(cudaEventElapsedTime(&time, start, stop));\
	CUDA_SAFE_CALL(cudaEventDestroy(start)); \
	CUDA_SAFE_CALL(cudaEventDestroy(stop));
	  
/** Initialize CUDA Runtime API.
 * There is no explicit init-function, the API is initialized with the first API call, 
 * the called functions are cudaDeviceReset and cudaFree.
 */
void initDevice(void);

// Parallel CUDA random graph generator 1, PER:
void PER_Generator(bool *content, 			/* in\out */ 
				   float skipping_prob, 	/* in */ 
				   int vertex_num, 		 	/* in */ 
				   int edges_num);		 	/* in */ 

// Parallel CUDA random graph generator 2, PZER:
void PZER_Generator(bool *content,   		/* in\out */ 
				   	float skipping_prob,	/* in */ 
				   	int lambda, 			/* in */ 
				   	int vertex_num, 		/* in */ 
				   	int edges_num);			/* in */ 

// Parallel CUDA random graph generator 3, PPreZER:
void PPreZER_Generator(bool *content, 		/* in\out */ 
					   float skipping_prob, /* in */ 
					   int lambda, 			/* in */ 
					   int m, 				/* in */ 
					   int vertex_num, 		/* in */ 
				   	   int edges_num);		/* in */ 

#endif










