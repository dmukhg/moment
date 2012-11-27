#ifndef MOMENT_NEURON_CUH

#define MOMENT_NEURON_CUH

#include <stdlib.h> // For rand()

#include "defs.cuh"

// Structure representing a connection. Has the id of the post
// synaptic neuron, the synaptic weight and the index of the next
// connection
struct Connection {
  int neuron;
  float weight;
  unsigned int next;
};

/* Structure representing a neuron in the Izhikewich model.
 *
 * current represents Thalamic Input.
 * connection represents the first connection that this neuron has.
 */
struct Neuron {
  float current   // 'I' Input current. Sum of all input potential
      , potential // 'v' Membrane potential
      , recovery; // 'u' Negative feedback 

  unsigned int connection;

  Neuron() : current(0.0), potential(0.0), recovery(0.0), connection(0) {}
};

// Initialization functions
void input_random_current(Neuron *neurons);

// Per-iteration functions
__global__ void update_potential(Neuron *neurons, 
    Connection *connections,int number);

#endif
