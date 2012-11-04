#ifndef MOMENT_CONNECTIONS

#define MOMENT_CONNECTIONS

#include <stdlib.h> // For rand()

#include "defs.cuh"

// Structure representing a connection. Has the id of the post
// synaptic neuron and the axonal delay
struct Connection {
  int neuron;
  int delay;
};

/* Structure representing a neuron in the Izhikewich model.
 *
 * current represents Thalamic Input.
 */
struct Neuron {
  float current   // 'I' Input current. Sum of all input potential
      , potential // 'v' Membrane potential
      , recovery; // 'u' Negative feedback 

  Neuron() : current(0.0), potential(0.0), recovery(0.0) {}
};

// Initialization functions
void input_random_current(Neuron *neurons);

// Per-iteration functions
__global__ void update_potential(Neuron *neurons, int number);

#endif
