#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <assert.h>

#include "network.cuh"

void test_nutrients(Network *n) {
  char name[23], line[LINE_MAX];
  int energy, protein, fat, calcium;
  float iron;
  Neuron *neu;
  int *rate;

  while (fgets(line, LINE_MAX, stdin) != NULL) {
    sscanf(line, "%23c%d%d%d%d%f", &name, &energy, &protein,
        &fat, &calcium, &iron);

    // Reset inputs 
    neu = n->neurons(true);
    neu[0].input = 0;
    neu[1].input = 0;
    neu[2].input = 0;
    neu[3].input = 0;
    neu[4].input = 0;
   
    for (int i=0; i<1000; i++) {
      if (i == 100) {
        // Allow time for network to stabilize
        neu = n->neurons(true);
        neu[0].input = energy * 1.0;
        neu[1].input = protein * 1.0;
        neu[2].input = fat* 1.0;
        neu[3].input = calcium * 0.01;
        neu[4].input = iron;

        n->neurons(neu, true);
      }
      
      n->time_step();
      n->find_firing_neurons();
      n->update_current();
      n->update_potential();
    }

    rate = n->spiking_rate(true);
    neu = n->neurons(true);
    printf("%d, %d\n", rate[11], rate[12]);
    //printf(">>%d, %d\n", neu[11].current, neu[12].current);
  };
}

int main() {
  Network *n = new Network(13, 42);
  n->randomize_weights();
  n->build_connections(5,6,2);

  test_nutrients(n);
}
