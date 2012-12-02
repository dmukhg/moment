#include "iteration.cuh"

/* Increment the time-step by 1.  
 * For use only with a single block and a single thread */
__global__ void time_step(int *dev_time)
{
    *dev_time += 1;
}

/* For all neurons with potential > IzTHRESHOLD, add their indices to
 * the fired array. Also reset the neurons which are firing. */
__global__ void find_firing_neurons(Neuron *neurons,
    bool *fired, int number)
{
  int offset = blockIdx.x * blockDim.x * blockDim.y + threadIdx.x;

  float v, u;

  if (offset >= number) {
    // There are no such neurons
    return;
  }

  v = neurons[offset].potential;
  u = neurons[offset].recovery;

  if (v > IzTHRESHOLD) {
    fired[offset] = true;

    // reset firing neuruons
    neurons[offset].potential = IzC;
    neurons[offset].recovery  = u + IzD;
  }
}

/* For all fired neurons, update the thalamic input on connected
 * neurons.*/
__global__ void update_current(Neuron *neurons, Connection *connections,
    bool *fired, int number) {
  int offset = blockIdx.x * blockDim.x * blockDim.y + threadIdx.x;
  int cIdx, nIdx;

  // First, make the current on all neurons 0 + the input value for
  // that neuron
  if (offset < number) {
    neurons[offset].current = 0 + 1000*neurons[offset].input;
  }
  // Ensure that *all* neurons have 0 current
  __syncthreads();

  if (fired == NULL || fired[offset] == false) {
    // No such fired neuron
    return;
  }

  cIdx    = neurons[offset].connection;

  if (connections != NULL) {
    do {
      nIdx = connections[cIdx].neuron;
      atomicAdd(&(neurons[nIdx].current), 1000*connections[cIdx].weight);
     // atomicAdd(&neurons[connections[cIdx].neuron].current,
      //    1.0f);
      cIdx = connections[cIdx].next;
    } while (cIdx >= 0);
  }
}
