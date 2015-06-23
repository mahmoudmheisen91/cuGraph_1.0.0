#ifndef PARALLEL_FUNCTIONS_H_
#define PARALLEL_FUNCTIONS_H_

#include <ctime>
#include <cstdlib>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

void initDevice(void);
void parallel_PER(bool *content, float p, int V, int E);
void parallel_PZER(bool *content, float p, int lambda, int V, int E);
void parallel_PPreZER(bool *content, float p, int lambda, int m, int V, int E);

#endif
