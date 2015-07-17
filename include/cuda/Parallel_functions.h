#ifndef PARALLEL_FUNCTIONS_H_
#define PARALLEL_FUNCTIONS_H_

#include <ctime>
#include <cstdlib>

void initDevice(void);
void parallel_PER(bool *content, float p, int V, int E);
void parallel_PZER(bool *content, float p, int lambda, int V, int E);
void parallel_PPreZER(bool *content, float p, int lambda, int m, int V, int E);

#endif
