#ifndef PARALLEL_FUNCTIONS_H_
#define PARALLEL_FUNCTIONS_H_

#define BLOCK_SIZE 256

#define ZERO_BANK_CONFLICTS 
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

#ifdef ZERO_BANK_CONFLICTS
	#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS + (index) >> (2*LOG_NUM_BANKS))
#else
	#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)
#endif

#include <ctime>

void preallocBlockSums(unsigned int maxNumElements);
void prescanArray(float *outArray, float *inArray, int numElements);
void deallocBlockSums();

void parallel_scan(int num_elements, float* h_data);
void parallel_generateRandomNumber(float *R, int B);
void parallel_generateSkipValue(int *S, float *R, int B, float p);

#endif
