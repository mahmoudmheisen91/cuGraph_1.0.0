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
#include <cstdlib>
#include<curand.h>
#include<curand_kernel.h>

void initDevice(void);
void preallocBlockSums(unsigned int maxNumElements);
void prescanArray(float *outArray, float *inArray, int numElements);
void deallocBlockSums();

void parallel_PER(bool *content, float p, int V, int E);
void parallel_PZER(bool *content, float p, int lambda, int V, int E);
void parallel_PPreZER(bool *content, float p, int lambda, int m, int V, int E);

#endif
