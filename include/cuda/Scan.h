#ifndef _SCAN_CU_
#define _SCAN_CU_

#define BLOCK_SIZE 256

#define ZERO_BANK_CONFLICTS 
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

#ifdef ZERO_BANK_CONFLICTS
	#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS + (index) >> (2*LOG_NUM_BANKS))
#else
	#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)
#endif

void scan(int num_elements, float* h_data);

#endif
