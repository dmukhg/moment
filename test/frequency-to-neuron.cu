/* Test the feeding of a single frequency time-series as an input to a single
 * neuron. */

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include "neuron.cu"
#include "iteration.cu"

#define WEIGHT 20.0

int test_frequency_in_single_neuron(void)
{
  int *dev_time;
  int host_time = 0;
  bool *dev_fired, host_fired[1];

  char line[LINE_MAX];
  float host_value;

  Neuron host_neurons[1];
  Neuron *dev_neurons;

  // Allocate memory on the GPU
  cudaMalloc( (void**)&dev_time, sizeof(int));
  cudaMalloc( (void**)&dev_neurons, sizeof(Neuron));

  // Initialization
  host_neurons[0].current=0;
  fill_false(host_fired, 1);
  cudaMemcpy(dev_time, &host_time, sizeof(int),
      cudaMemcpyHostToDevice);
  cudaMemcpy(dev_neurons, &host_neurons, sizeof(Neuron),
      cudaMemcpyHostToDevice);

  while (fgets(line, LINE_MAX, stdin) != NULL) {
    time_step<<<1,1>>>(dev_time);
    cudaMalloc( (void**)&dev_fired, sizeof(bool));
    cudaMemcpy(dev_fired, &host_fired, sizeof(bool), cudaMemcpyHostToDevice);

    host_value = strtof(line, NULL);

    host_neurons[0].input = WEIGHT * host_value;
    cudaMemcpy(dev_neurons, &host_neurons, sizeof(Neuron),
        cudaMemcpyHostToDevice);

    find_firing_neurons<<<1,1>>>(dev_neurons, dev_fired, 1);
    update_current<<<1,1>>>(dev_neurons, NULL, dev_fired, 1);
    update_potential<<<1,1>>>(dev_neurons, NULL, 1);

    cudaMemcpy(host_neurons, dev_neurons, sizeof(Neuron),
        cudaMemcpyDeviceToHost);

    printf("[%d, %10f],\n", host_time, host_neurons[0].potential);

    host_time++;
  }

  return 0;
}

int main(void)
{
  test_frequency_in_single_neuron();
  return 0;
}
