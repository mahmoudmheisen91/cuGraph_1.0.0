#include "mainPPS.hpp"

void test_scan(int size) {
	// declare host and device variable:
	int *host_in, *host_out, *dev_in, *dev_out, n = size;
    int *test_out, *mid1, *mid1_out, *mid2, *mid2_out;
	float time1;
	cudaEvent_t start, stop; 

	// allocate host memory:
	host_in = new int[n];
	mid1 = new int[4];
	mid1_out = new int[4];
	host_out = new int[n];
    test_out = new int[n];
	
	// allocate device memory:
	cudaMalloc((void**)&dev_in, n * sizeof(int));
	cudaMalloc((void**)&dev_out, n * sizeof(int));
	cudaMalloc((void**)&mid2_out, 4 * sizeof(int));
	cudaMalloc((void**)&mid2, 4 * sizeof(int));
	
	// fill 
	for(int i = 0; i < n; i++) {
		host_in[i] = i+1;
        test_out[i] = i+1;
	}
	
	// copy host vars to device vars:
	cudaMemcpy(dev_in, host_in, n*sizeof(int), cudaMemcpyHostToDevice);

    // start Device kernal:
	cudaEventCreate(&start); 
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0); 

	scanGlobal<<<4, 256>>>(dev_out, dev_in, 256);
	cudaMemcpy(host_out, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);
	/*mid1[0] = host_out[255];
	mid1[1] = host_out[511];
	mid1[2] = host_out[511+256];
	mid1[3] = host_out[1023];
	cudaMemcpy(mid2, mid1, 4*sizeof(int), cudaMemcpyHostToDevice);
	scanGlobal<<<1, 4>>>(mid2_out, mid2, 4);
	cudaMemcpy(mid1_out, mid2_out, 4 * sizeof(int), cudaMemcpyDeviceToHost);
	
	for(int i = 256; i < 512; i++) {
       //host_out[i] += mid1_out[0];
	}
	
	for(int i = 512; i < 512 + 256; i++) {
       host_out[i] += mid1_out[1];
	}
	
	for(int i = 512 + 256; i < 1024; i++) {
       host_out[i] += mid1_out[2];
	}*/

	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop);  
	cudaEventElapsedTime(&time1, start, stop); 
	cudaEventDestroy(start); 
	cudaEventDestroy(stop);
	
	// print time:
	cout << endl << "Kernal time = " << time1 << " ms, size = " << n << endl;
	
	// copy device vars to host vars:
	//cudaMemcpy(host_out, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);
	
    exclusive_scan_sum(test_out, size);

	for(int i = 0; i < 512; i++) {
        //if(test_out[i] != host_out[i])
            cout << host_out[i] << " " ;
	}

	// free host memory:
	delete host_in;
	delete host_out;
    delete test_out;
	
	// free device memory:
	cudaFree(dev_in);
	cudaFree(dev_out);
}
