#include <cuda/Parallel_functions.h> 

__global__ void skipValue_kernal(int *S, float *R, int *B, float *p);

void parallel_generateSkipValue(int *S, float *R, int B, float p) {
	int *d_S, *d_B, *h_B;
	float *d_R, *h_p, *d_p;
	
	// allocate:
	h_B = new int[1];
	h_p = new float[1];
	cudaMalloc((void**) &d_S, B * sizeof(int));
	cudaMalloc((void**) &d_B, sizeof(int));
	cudaMalloc((void**) &d_R, B * sizeof(float));
	cudaMalloc((void**) &d_p, sizeof(float));
	
	// copy:
	h_B[0] = B;
	h_p[0] = p;
	cudaMemcpy(d_B, h_B, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_p, h_p, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_R, R, B * sizeof(float), cudaMemcpyHostToDevice);
	
	// run kernel:
	skipValue_kernal<<<pow(2, 16)-1, pow(2, 10)>>>(d_S, d_R, d_B, d_p);
	
	// copy:
	cudaMemcpy(S, d_S, B * sizeof(float), cudaMemcpyDeviceToHost);
	
	// free:
	delete h_B;
	delete h_p;
	cudaFree(d_S);
	cudaFree(d_B);
	cudaFree(d_R);
	cudaFree(d_p);
}
