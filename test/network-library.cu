#include <stdio.h>
#include <assert.h>

#include "network.cuh"

void test_time_step(Network n) {
  int before = n.time();
  n.time_step();
  int after = n.time();

  assert((before + 1) == after);
  printf(".");

  return;
}

void test_accessors(Network n) {
  Neuron * neurons = n.neurons();
  Connection * cons = n.connections();

  // Set values
  neurons[0].potential = 11.0;
  // Copy values and sync
  n.neurons(neurons, true);
  // Copy back after sync
  neurons = n.neurons(true);
  assert(neurons[0].potential == 11.0);
  printf(".");

  cons[0].neuron = 12;
  n.connections(cons, true);
  cons = n.connections();
  assert(cons[0].neuron == 12);
  printf(".");
  
  return;
}

int main() {
  Network n = Network(2, 1);

  test_time_step(n);
  test_accessors(n);

  printf("\n");
  return 0;
}
