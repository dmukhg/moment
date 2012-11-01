#include "connections.cuh"

/* Establishes a random value of synaptic current on the input
 * neurons. */
void input_random_current(Neuron *neurons) 
{
  int i, r;

  for (i=0; i < INNEURON; i++) {
    r = rand() % 10; // Assumes seeding has been done

    neurons[i].current = (r > 5) ? 40 : 0;
  }
}
