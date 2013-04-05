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
  Network *n = new Network(3, 2);

  Neuron *neu = n->neurons();
  Connection *c = n->connections();

  n->build_connections(1,1,1);

  test_kernel_find_firing(n);

  return 0;
}
