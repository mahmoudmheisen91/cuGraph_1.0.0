#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <iostream>

// includes, kernels
#include "scan.h" 

int main() {  
	int num_elements = 16777218;
	float* h_data = (float*) malloc( sizeof( float) * num_elements); 
	
	// initialize the input data on the host
    for( unsigned int i = 0; i < num_elements; ++i) h_data[i] = 1.0f;
	
    parallel_scan(num_elements, h_data);
    std::cout << h_data[0] << h_data[num_elements-1] << std::endl;
    free(h_data);
}
