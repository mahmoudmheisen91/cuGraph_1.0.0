#include <cuda/Parallel_functions.h> 
#include <iostream>

__global__ void random_number_generator_kernal(int *masterSeed, int *size, float *PRNG);
__global__ void skipValue_kernal(float *S, float *R, int *B, float *p);
__global__ void skipValuePre_kernal(float *S, float *R, int *B, float *p, int *m, float *F);
__global__ void addEdges_kernal(bool *content, float *S, int *V, int *B, float *L, float *last);

void initDevice(void) {
	cudaFree(0);
}

void parallel_PZER(bool *content, float p, int lambda, int V, int E) {
	// declerations:
	bool *d_content;
	float *d_R, *d_S, *d_p, *h_p, *d_odata, *d_L, *h_L, *h_last, *d_last, *h_S;
	int *d_seed, *h_seed, *d_B, *h_B, *d_V, *h_V;
	
	int B, L = 0;
    int seed = time(0)-1000000000;
    double segma = sqrt(p * (1 - p) * E);

    if((int)(p * E + lambda * segma) < 2000000)
        B = (int)(p * E + lambda * segma);
    else
        B = 2000000;

	// allocation:
	h_p = new float[1];
	h_seed = new int[1];
	h_V = new int[1];
	h_B = new int[1];
	h_L = new float[1];
	h_last = new float[1];
	h_S = new float[B];
	cudaMalloc((void**) &d_p, sizeof(float));
	cudaMalloc((void**) &d_seed, sizeof(int));
	cudaMalloc((void**) &d_V, sizeof(int));
	cudaMalloc((void**) &d_B, sizeof(int));
	cudaMalloc((void**) &d_L, sizeof(float));
	cudaMalloc((void**) &d_last, sizeof(int));
	cudaMalloc((void**) &d_content, V * V * sizeof(bool)); 	// 100 MB
	cudaMalloc((void**) &d_R, B * sizeof(float)); 			// 8 MB 
	cudaMalloc((void**) &d_S, B * sizeof(float));			// 8 MB
	cudaMalloc((void**) &d_odata, B * sizeof(float));		// 8 MB
	
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
	
	srand(time(0));
	h_last[0] = 0;
	
	// run kernals:
	while(L < E) {
		cudaMemcpy(d_last, h_last, sizeof(float), cudaMemcpyHostToDevice);
		
		random_number_generator_kernal<<<1, pow(2, 10)>>>(d_seed, d_B, d_R);
		skipValue_kernal<<<32, pow(2, 10)>>>(d_S, d_R, d_B, d_p);	
		preallocBlockSums(B);
		prescanArray(d_odata, d_S, B);
		addEdges_kernal<<<32, pow(2, 10)>>>(d_content, d_odata, d_V, d_B, d_L, d_last);
		
		// copy:
		cudaMemcpy(h_L, d_L, sizeof(float), cudaMemcpyDeviceToHost);
		L = (int)h_L[0];
		
		cudaMemcpy(h_S, d_odata, B*sizeof(float), cudaMemcpyDeviceToHost);
		h_last[0] = h_S[B-1];
		//std::cout << h_last[0] << std::endl;
	}
	
	cudaMemcpy(content, d_content, sizeof(bool) * V * V, cudaMemcpyDeviceToHost);
	
	// free:
	delete h_p;
	delete h_seed;
	delete h_B;
	delete h_V;
	delete h_L;
	delete h_last;
	delete h_S;
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
    cudaFree(d_last);
}

void parallel_PPreZER(bool *content, float p, int lambda, int m, int V, int E) {
	// declerations:
	bool *d_content;
	float *d_R, *d_S, *d_p, *h_p, *d_odata, *d_L, *h_L, *h_last, *d_last, *h_S, *h_F, *d_F;
	int *d_seed, *h_seed, *d_B, *h_B, *d_V, *h_V, *d_m, *h_m;
	
	int B, L = 0;
    int seed = time(0)-1000000000;
    double segma = sqrt(p * (1 - p) * E);

    if((int)(p * E + lambda * segma) < 2000000)
        B = (int)(p * E + lambda * segma);
    else
        B = 2000000;

	float *F = new float[m+1];
    for(int i = 0; i <= m; i++) {
        F[i] = 1 - pow(1-p, i+1);
    }
        
	// allocation:
	h_p = new float[1];
	h_seed = new int[1];
	h_V = new int[1];
	h_B = new int[1];
	h_L = new float[1];
	h_last = new float[1];
	h_S = new float[B];
	h_F = new float[m+1];
	h_m = new int[1];
	cudaMalloc((void**) &d_p, sizeof(float));
	cudaMalloc((void**) &d_seed, sizeof(int));
	cudaMalloc((void**) &d_V, sizeof(int));
	cudaMalloc((void**) &d_B, sizeof(int));
	cudaMalloc((void**) &d_L, sizeof(float));
	cudaMalloc((void**) &d_last, sizeof(int));
	cudaMalloc((void**) &d_m, sizeof(int));
	cudaMalloc((void**) &d_F, (m+1) * sizeof(float));
	cudaMalloc((void**) &d_content, V * V * sizeof(bool)); 	// 100 MB
	cudaMalloc((void**) &d_R, B * sizeof(float)); 			// 8 MB 
	cudaMalloc((void**) &d_S, B * sizeof(float));			// 8 MB
	cudaMalloc((void**) &d_odata, B * sizeof(float));		// 8 MB
	
	// fill:
	h_p[0] = p;
	h_seed[0] = seed;
	h_V[0] = V;
	h_B[0] = B;
	h_m[0] = m;
	
	// copy:
	cudaMemcpy(d_p, h_p, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_seed, h_seed, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V, h_V, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_L, h_L, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_F, h_F, (m+1) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_m, h_m, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_content, content, V * V * sizeof(bool), cudaMemcpyHostToDevice);
	
	srand(time(0));
	h_last[0] = 0;
	
	// run kernals:
	while(L < E) {
		cudaMemcpy(d_last, h_last, sizeof(float), cudaMemcpyHostToDevice);
		
		random_number_generator_kernal<<<1, pow(2, 10)>>>(d_seed, d_B, d_R);
		skipValuePre_kernal<<<32, pow(2, 10)>>>(d_S, d_R, d_B, d_p, d_m, d_F);	
		preallocBlockSums(B);
		prescanArray(d_odata, d_S, B);
		addEdges_kernal<<<32, pow(2, 10)>>>(d_content, d_odata, d_V, d_B, d_L, d_last);
		
		// copy:
		cudaMemcpy(h_L, d_L, sizeof(float), cudaMemcpyDeviceToHost);
		L = (int)h_L[0];
		
		cudaMemcpy(h_S, d_odata, B*sizeof(float), cudaMemcpyDeviceToHost);
		h_last[0] = h_S[B-1];
		//std::cout << h_last[0] << std::endl;
	}
	
	cudaMemcpy(content, d_content, sizeof(bool) * V * V, cudaMemcpyDeviceToHost);
	
	// free:
	delete h_p;
	delete h_seed;
	delete h_B;
	delete h_V;
	delete h_L;
	delete h_last;
	delete h_S;
	delete h_m;
	delete h_F;
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
    cudaFree(d_last);
    cudaFree(d_m);
    cudaFree(h_F);
}

void parallel_PER(bool *content, float p, int V, int E) {

}










