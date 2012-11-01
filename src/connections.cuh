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
 * Although the text says current, we are going for a slightly
 * different approach. Current is actually a potential value, that of
 * the input channels combined. In the Izhikewich model, the value is
 * named current and because junctions are able to linearly add
 * currents via Kirchoff's Law. We are assuming the current has been
 * added linearly and then has generated an input synaptic potential
 * in mV and we are still calling it current.
 */
struct Neuron {
  int current   // 'I' Input current. Sum of all input potential
    , potential // 'v' Membrane potential
    , recovery; // 'u' Negative feedback 
};

void input_random_current(Neuron *neurons);
#endif
