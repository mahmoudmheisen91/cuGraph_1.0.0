#include <cuda/Parallel_functions.h> 

void parallel_scan(float* S, int B) {   
    // allocate device memory input and output arrays
    float* d_idata = NULL;
    float* d_odata = NULL;
    cudaMalloc( (void**) &d_idata, sizeof(float) * B);
    cudaMalloc( (void**) &d_odata, sizeof(float) * B);
    
    // copy host memory to device input array
    cudaMemcpy( d_idata, S, sizeof(float) * B, cudaMemcpyHostToDevice);
    
    // run 
    preallocBlockSums(B);
    prescanArray(d_odata, d_idata, B);
    
    // copy device memory to host input array
    cudaMemcpy(S, d_odata, sizeof(float) * B, cudaMemcpyDeviceToHost);
    
    deallocBlockSums();
    cudaFree(d_odata);
    cudaFree(d_idata);
}

