/* Kernels that the SNN uses. Paritioning of tasks as defined in Nageshwaran */
#ifndef MOMENT_KERNELS_CUH

#define MOMENT_KERNELS_CUH

#include "types.cuh"

__global__ void _time_step(int *dev_time) {
  /* Increment the time value in the device by 1 */
  *dev_time += 1;
};

__global__ void find_firing_neurons(Neuron *neurons, int *dev_time, 
    bool *fired, int *rate, bool *fired_table, int number, int fired_res)
{
  int offset = blockIdx.x * blockDim.x * blockDim.y + threadIdx.x;
  int table_offset = *dev_time % fired_res;
};

#endif
