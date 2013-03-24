#include <stdio.h>
#include <assert.h>

#include "network.cuh"

void test_kernel_find_firing(Network *n) {
  Neuron *neu;

  for (int i=0; i<1000; i++) {
    neu = n->neurons(true);
    printf("%f\n", neu[1].potential);

    if (i == 100) {
      neu = n->neurons(true);
      neu[0].input = 12.0;
      n->neurons(neu, true);
    }

    n->time_step();
    n->find_firing_neurons();
    n->update_current();
    n->update_potential();
  }
}

int main() {
  Network *n = new Network(2, 1);

  Neuron *neu = n->neurons();
  Connection *c = n->connections();

  neu[0].connection = 0;
  c[0].next = -1;
  c[0].weight = 20.0;
  c[0].neuron = 1;

  neu[1].connection = -1;

  n->neurons(neu, true);
  n->connections(c, true);


  test_kernel_find_firing(n);

  return 0;
}
