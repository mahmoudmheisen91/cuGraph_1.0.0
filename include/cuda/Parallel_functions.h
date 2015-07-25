/*
 * Parallel_functions.h
 *
 *  Created: 2015-05-01, Modified: 2015-07-25
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

/** Initialize CUDA Runtime API.
 * There is no explicit init-function, the API is initialized with the first API call, 
 * the called functions are cudaDeviceReset anf cudaFree.
 */
void initDevice(void);

// Parallel CUDA random graph generator 1, PER:
void parallel_PER(bool *content, 	/* in\out */ 
				  float p, 		 	/* in */ 
				  int V, 		 	/* in */ 
				  int E);		 	/* in */ 

// Parallel CUDA random graph generator 2, PZER:
void parallel_PZER(bool *content,   /* in\out */ 
				   float p, 		/* in */ 
				   int lambda, 		/* in */ 
				   int V, 			/* in */ 
				   int E);			/* in */ 

// Parallel CUDA random graph generator 3, PPreZER:
void parallel_PPreZER(bool *content, 	/* in\out */ 
					  float p, 			/* in */ 
					  int lambda, 		/* in */ 
					  int m, 			/* in */ 
					  int V, 			/* in */ 
					  int E);			/* in */ 

#endif










