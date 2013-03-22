/* Kernels that the SNN uses. Paritioning of tasks as defined in Nageshwaran */
#ifndef MOMENT_KERNELS_CUH

#define MOMENT_KERNELS_CUH

#include "types.cuh"
#include "defs.cuh"

__global__ void _time_step(int *dev_time) {
  /* Increment the time value in the device by 1 */
  *dev_time += 1;
};

__global__ void _find_firing_neurons(Neuron *neurons, int *dev_time, 
    bool *fired, int *rate, bool *fired_table, int n_neurons, int fired_res)
{
  int offset = blockIdx.x * blockDim.x * blockDim.y + threadIdx.x;
  int table_offset = *dev_time % fired_res;

  float v, u;

  if (offset >= n_neurons) {
    // There are no such neurons
    return;
  }

  v = neurons[offset].potential;
  u = neurons[offset].recovery;

  if (v > IzTHRESHOLD) {
    // Should fire
    fired[offset] = true;

    // reset the neuron
    neurons[offset].potential = IzC;
    neurons[offset].recovery = u + IzD;

    // Add this firing to the firing table
    // Compute spike firing rate
    if (!fired_table[fired_res * offset + table_offset]) {
      // If there wasn't a spike at this offset, increment spike rate.
      // If there was, doesn't really matter.
      rate[offset] += 1;
    }

    fired_table[fired_res * offset + table_offset] = true;
  } else {
    // Spike rate adjustments

    // If there was a spike at this offset, decrement spike rate.
    // If there wasn't, doesn't matter
    if (fired_table[fired_res * offset + table_offset]) {
      rate[offset] -= 1;
      fired_table[fired_res * offset + table_offset] = false;
    }
  }
};

#endif
