__device__ int single_warp_scan(int *data, int idx);
__device__ int single_block_scan(int *data, int idx);
__global__ void global_scan_kernel_1(int *data, int *block_results);
__global__ void global_scan_kernel_2(int *block_results);
__global__ void global_scan_kernel_3(int *data, int *block_results);
