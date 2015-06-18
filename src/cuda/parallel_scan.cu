#include <cuda/Parallel_functions.h> 

void parallel_scan(int num_elements, float* h_data) {   
    // allocate device memory input and output arrays
    float* d_idata = NULL;
    float* d_odata = NULL;
    cudaMalloc( (void**) &d_idata, sizeof(float) * num_elements);
    cudaMalloc( (void**) &d_odata, sizeof(float) * num_elements);
    
    // copy host memory to device input array
    cudaMemcpy( d_idata, h_data, sizeof(float) * num_elements, cudaMemcpyHostToDevice);
    
    // run 
    preallocBlockSums(num_elements);
    prescanArray(d_odata, d_idata, num_elements);
    
    // copy device memory to host input array
    cudaMemcpy(h_data, d_odata, sizeof(float) * num_elements, cudaMemcpyDeviceToHost);
    
    deallocBlockSums();
    cudaFree(d_odata);
    cudaFree(d_idata);
}

