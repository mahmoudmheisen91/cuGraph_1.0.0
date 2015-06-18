#include <cuda/Parallel_functions.h> 
#include <iostream>

__global__ void random_number_generator_kernal(int *masterSeed, int *size, float *PRNG);
__global__ void skipValue_kernal(float *S, float *R, int *B, float *p);
__global__ void addEdges_kernal(bool *content, float *S, int *V, int *B, float *L);

void parallel_PZER(bool *content, float p, int lambda, int V, int E, int &numberOfEdges) {
	// declerations:
	bool *d_content;
	float *d_R, *d_S, *d_p, *h_p, *d_odata, *d_L, *h_L;
	int *d_seed, *h_seed, *d_B, *h_B, *d_V, *h_V;
	
	int B, L = 0;
    int seed = time(0)-1000000000;
    double segma = sqrt(p * (1 - p) * E);

    if((int)(p * E + lambda * segma) < 1000000)
        B = (int)(p * E + lambda * segma);
    else
        B = 1000000;

	// allocation:
	h_p = new float[1];
	h_seed = new int[1];
	h_V = new int[1];
	h_B = new int[1];
	h_L = new float[1];
	cudaMalloc((void**) &d_p, sizeof(float));
	cudaMalloc((void**) &d_seed, sizeof(int));
	cudaMalloc((void**) &d_V, sizeof(int));
	cudaMalloc((void**) &d_B, sizeof(int));
	cudaMalloc((void**) &d_L, sizeof(float));
	cudaMalloc((void**) &d_content, V * V * sizeof(bool));
	cudaMalloc((void**) &d_R, B * sizeof(float));
	cudaMalloc((void**) &d_S, B * sizeof(float));
	cudaMalloc((void**) &d_odata, sizeof(float) * B);
	
	// fill:
	h_p[0] = p;
	h_seed[0] = seed;
	h_V[0] = V;
	h_B[0] = B;
	
	// copy:
	cudaMemcpy(d_p, h_p, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_seed, h_seed, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V, h_V, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_L, h_L, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_content, content, V * V * sizeof(bool), cudaMemcpyHostToDevice);
	
	// run kernals:
	while(L < p*E + lambda * segma) {
		random_number_generator_kernal<<<1, pow(2, 10)>>>(d_seed, d_B, d_R);
		skipValue_kernal<<<32, pow(2, 10)>>>(d_S, d_R, d_B, d_p);
		preallocBlockSums(B);
		prescanArray(d_odata, d_S, B);
		addEdges_kernal<<<32, pow(2, 10)>>>(d_content, d_odata, d_V, d_B, d_L);
		
		// copy:
		cudaMemcpy(h_L, d_L, sizeof(float), cudaMemcpyDeviceToHost);
		L += (int)h_L[0];
		numberOfEdges += B;
	}
	
	cudaMemcpy(content, d_content, sizeof(bool) * V * V, cudaMemcpyDeviceToHost);
	
	// free:
	delete h_p;
	delete h_seed;
	delete h_B;
	delete h_V;
	delete h_L;
	cudaFree(d_p);
	cudaFree(d_seed);
	cudaFree(d_B);
	cudaFree(d_V);
	cudaFree(d_L);
	cudaFree(d_content);
	cudaFree(d_R);
	cudaFree(d_S);	
	deallocBlockSums();
    cudaFree(d_odata);
}

void initDevice(void) {
	cudaFree(0);
}


















