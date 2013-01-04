/* Test variation of spiking output w.r.t. weight of input signal. */

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define WEIGHT 20.0

// Note that we are including the source file and not the header file only.
// This allows us access to static variables and globals (gasp!). This is more
// or less a close replica of the exposure that the functions being tested will
// get.
#include "neuron.cu"
#include "iteration.cu"

/* Study the spike pattern in a single neuron with an actual time-series of
 * vibration. For ease, assume the time-step of sampling of the vibration data
 * is the same as the resolution of the neuron.
 *
 * Input is to be provided with stdin. A set of values separated by newlines.
 *
 * The output is meant to be viewed with the included spiking visualizer in
 * test/visualizer/
 */
int test_vibration_in_single_neuron(void)
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
  host_neurons[0].current = 0;
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
  test_vibration_in_single_neuron();

  return 0;
}

