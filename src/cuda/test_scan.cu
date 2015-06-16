#include "mainPPS.hpp"

void test_scan(int size) {
	// declare host and device variable:
	int *host_in, *host_out, *dev_in, *dev_out, n = size;
    int *test_out;
	float time1;
	cudaEvent_t start, stop; 
	
	dim3 grid(1);
	dim3 block(n);

	// allocate host memory:
	host_in = new int[n];
	host_out = new int[n];
    test_out = new int[n];
	
	// allocate device memory:
	cudaMalloc((void**)&dev_in, n * sizeof(int));
	cudaMalloc((void**)&dev_out, n * sizeof(int));
	
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

	cuScan<<<grid, block, n * sizeof(float)>>>(dev_out, dev_in, n);

	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop);  
	cudaEventElapsedTime(&time1, start, stop); 
	cudaEventDestroy(start); 
	cudaEventDestroy(stop);
	
	// print time:
	cout << endl << "Kernal time = " << time1 << " ms, size = " << n << endl;
	
	// copy device vars to host vars:
	cudaMemcpy(host_out, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);
	
    // check outout:
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    exclusive_scan_sum(test_out, size);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time1, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // print time:
    cout << endl << "Host time = " << time1 << " ms, size = " << n << endl;

	for(int i = 0; i < n; i++) {
        if(test_out[i] != host_out[i])
            cout << test_out[i] << " " << host_out[i] << endl;
	}

	// free host memory:
	delete host_in;
	delete host_out;
    delete test_out;
	
	// free device memory:
	cudaFree(dev_in);
	cudaFree(dev_out);
}
