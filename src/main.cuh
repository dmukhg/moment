#ifndef MOMENT_MAIN

#define NEURONS_PER_LAYER  10000
#define LAYERS  6

// Post synaptic connections per neuron: should be clear enough though
#define POST_SYN_CONN_PER_NEURON  50

// At this time, we are limiting the execution of the network to a finite
// number of time steps.  In real applications, this is probably sub-optimal.
// We need to figure out a better limiting condition.
#define ITERATIONS 100000

#endif
