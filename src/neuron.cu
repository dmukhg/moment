#include "neuron.cuh"

/* Takes a boolean array and fills it with false values */
void fill_false(bool *array, int num) 
{
  int i;

  for (i=0; i < num; i++) {
    array[i] = false;
  }

  return;
}


/* Establishes a random value of synaptic current on the input
 * neurons. */
void input_random_current(Neuron *neurons) 
{
  int i, r;

  for (i=0; i < INNEURON; i++) {
    r = rand() % 10; // Assumes seeding has been done

    neurons[i].input = (r > 5) ? 5 : 0.0;
  }
}

/* Go through each neuron and update the membrane potential based on
 * the Izhikewich model. 

 * This is for a single iteration. */
__global__ void update_potential(Neuron *neurons, 
    Connection *connections, int number)
{
  float del_v, del_u, v, u, I;
              
  int offset = blockIdx.x * blockDim.x * blockDim.y + threadIdx.x;

  if (offset >= number) { 
    // There are no such neurons
    return;
  }

  v = neurons[offset].potential;
  u = neurons[offset].recovery;
  I = neurons[offset].current / 1000.0;

  if (v < IzTHRESHOLD) {
    del_v = 0.04f*v*v + 5.0f*v + 140.0f - u + I;
    del_u = IzA * ( IzB*v - u);

    // Multiply by IzINCREMENT in this case is equivalent to
    // multipying with dx in a Taylor series expansion
    neurons[offset].potential = v + del_v * IzINCREMENT;
    neurons[offset].recovery  = u + del_u * IzINCREMENT;
  }
}
