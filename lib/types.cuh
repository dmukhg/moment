#ifndef MOMENT_TYPES_CUH

#define MOMENT_TYPES_CUH

/* Structure representing a connection. Has the id of the post
 * synaptic neuron, the synaptic weight and the index of the next
 * connection.
 */
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


  signed int connection;

  Neuron() : current(0), potential(140.0), recovery(0.0), connection(-1), input(0) {}
};

#endif
