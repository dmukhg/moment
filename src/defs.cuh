#ifndef MOMENT_DEFS

#define MOMENT_DEFS

/* A note about definitions as used in this program.
  
  LAYCOUNT  : Layer Count
  NEUPERLAY : Neurons per layer.
  OUTNEURON : Number of output neurons
  INNEURON  : Number of input neurons
  PSYNCONN  : Number of post-synaptic connections per neuron

  The Network.
  ---

  There are INNEURON number of input neurons.  Each of these neurons
  are connected to arbitrarily chosen PSYNCONN number of neurons in
  the first layer of the LAYCOUNT nubmer of layers in the network.

  Each neuron in the first layer is connected to arbitrary PSYNCONN
  number of neurons in the second layer and so on till the final
  layer.

  Each neuron in the last number fans in to one of the final output
  neurons.  This is ordered.  The first NEUPERLAY / OUTNEURON connect
  to the first output neuron, the next NEUPERLAY / OUTNEURON connect
  to the second and so on.  Any rounding off errors go to the final
  neuron.


  Reason for the tiny names
  ---

  Although it was possible to give truly mnemonic names to the
  definitions, they would have been unwieldingly large and would have
  cause legibility problems in the code.

  Future me, I am sorry, but this needs to be done.  :(
 */

#define LAYCOUNT  6
#define NEUPERLAY 10000
#define OUTNEURON 12
#define INNEURON  12
#define PSYNCONN  50 

// Computed directive counts neurons with outgoing connection
#define NUMNEURON LAYCOUNT * NEUPERLAY + INNEURON

// At this time, we are limiting the execution of the network to a
// finite number of time steps.  In real applications, this is
// probably sub-optimal.  We need to figure out a better limiting
// condition.
#define ITERATIONS 100000

#endif
