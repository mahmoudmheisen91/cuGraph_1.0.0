#include <cuda/Parallel_functions.h> 

__global__ void addEdges_kernal(bool *content, float *S, int *V, int *B);

void parallel_addEdges(bool *content, float *S, int numberOfVertices, int B) {
	int *d_B, *h_B, *d_V, *h_V;
	float *d_S;
	bool *d_content;
	
	// allocate:
	h_B = new int[1];
	h_V = new int[1];
	cudaMalloc((void**) &d_B, sizeof(int));
	cudaMalloc((void**) &d_V, sizeof(int));
	
	cudaMalloc((void**) &d_S, B * sizeof(float));
	cudaMalloc((void**) &d_content, numberOfVertices * numberOfVertices * sizeof(bool));
	
	// copy:
	h_B[0] = B;
	h_V[0] = numberOfVertices;
	cudaMemcpy(d_B, h_B, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V, h_V, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_S, S, B * sizeof(float), cudaMemcpyHostToDevice);
	
	// run kernel:
	addEdges_kernal<<<pow(2, 16)-1, pow(2, 10)>>>(d_content, d_S, d_V, d_B);
	
	// copy:
	cudaMemcpy(content, d_content, numberOfVertices * numberOfVertices * sizeof(bool), cudaMemcpyDeviceToHost);
	
	// free:
	delete h_B;
	delete h_V;
	cudaFree(d_S);
	cudaFree(d_B);
	cudaFree(d_content);
	cudaFree(d_V);
}
