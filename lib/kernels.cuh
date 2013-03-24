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

    // reset firing neuruons
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

__global__ void _update_current(Neuron *neurons, Connection *connections,
      bool *fired, int n_neurons)
{ 
  int offset = blockIdx.x * blockDim.x * blockDim.y + threadIdx.x;
  int cIdx, nIdx;

  // First make sure that the current on all neurons is 0 + the input value for
  // that neuron. */
  if (offset < n_neurons) {
    neurons[offset].current = 0 + 1000 * neurons[offset].input;
  }
  
  // Ensure that *all* neurons have 0 curent
  __syncthreads();

  if (fired[offset] == false) { // Neuron didn't fire
    return;
  }

  cIdx = neurons[offset].connection;

  if (connections != NULL) { // Traverse through all the connections of neuron
    while (cIdx != -1) {
      nIdx = connections[cIdx].neuron;
      atomicAdd(&(neurons[nIdx].current), 1000 * connections[cIdx].weight);
      cIdx = connections[cIdx].next;
    }; 
  }
}


__global__ void _update_potential(Neuron *neurons, Connection *connections, int
    n_neurons)
{
  float del_v, del_u, v, u, I;

  int offset = blockIdx.x * blockDim.x * blockDim.y + threadIdx.x;

  if (offset >= n_neurons) { // no such neuron
    return;
  }

  v = neurons[offset].potential;
  u = neurons[offset].recovery;
  I = neurons[offset].current / 1000.0;

  if (v < IzTHRESHOLD) {
    del_v = 0.04f*v*v + 5.0f*v + 140.0f - u + I;
    del_u = IzA * (IzB*v - u);

    // Multiply by IzINCREMENT in this case is equivalent to multiplying with
    // dx in a Taylor series expansion.
    neurons[offset].potential = v + del_v * IzINCREMENT;
    neurons[offset].recovery  = u + del_u * IzINCREMENT;
  }
}

#endif
