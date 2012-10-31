#ifndef MOMENT_UTILS

#define MOMENT_UTILS

#include "defs.cuh"

int get_index(int neuron, int connection);
__device__ int dev_get_index(int neuron, int connection);

#endif
