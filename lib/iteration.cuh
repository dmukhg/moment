#ifndef MOMENT_ITERATION_CUH
#define MOMENT_ITERATION_CUH
#define FIRED_RES 100

#include "defs.cuh"

__global__ void time_step(int *dev_time);
__global__ void find_firing_neurons(Neuron *neurons, int *dev_time,
    bool *fired, int *rate, bool *fired_queue, int number);
//__global__ void reset_firing_neurons(Neuron *neurons,
 //     int *fired, int *fIdx);
__global__ void update_current(Neuron *neurons, Connection *connections,
    bool *fired, int number);


#endif
