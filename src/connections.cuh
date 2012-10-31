#ifndef MOMENT_CONNECTIONS

#define MOMENT_CONNECTIONS

// Structure representing a connection. Has the id of the post
// synaptic neuron and the axonal delay
struct Connection {
  int neuron;
  int delay;
};

#endif
