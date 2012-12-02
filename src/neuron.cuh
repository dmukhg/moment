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
  int next;
};

/* Structure representing a neuron in the Izhikewich model.
 *
 * current represents Thalamic Input.
 * connection represents the first connection that this neuron has.
 */
struct Neuron {
  float potential // 'v' Membrane potential
      , recovery  // 'u' Negative feedback 
      , input;    // In case this is an input neuron

  int current; // 'I' Input current. Sum of all input potential


  unsigned int connection;

  Neuron() : current(0), potential(0.0), recovery(0.0), connection(0), input(0.0) {}
};

// Initialization functions
void input_random_current(Neuron *neurons);
void fill_false(bool *array, int num);

// Per-iteration functions
__global__ void update_potential(Neuron *neurons, 
    Connection *connections,int number);

#endif
