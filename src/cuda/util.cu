/*
 * util.cu
 *
 *  Created: 2015-07-26, Modified: 
 *
 */

// Headers includes:
#include <cuda/Cuda_Prototypes_Macros.h>

/** Initialize CUDA Runtime API.
 * There is no explicit init-function, the API is initialized with the first API call, 
 * the called functions are cudaDeviceReset and cudaFree.
 */
void initDevice(void) {
	CUDA_SAFE_CALL(cudaDeviceReset());
    CUDA_SAFE_CALL(cudaFree(0));
}












