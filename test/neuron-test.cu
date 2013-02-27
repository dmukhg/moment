/* Test neuron functionality */
#define NEU_COUNT 1

#include <stdio.h>

// Note that we are including the source file and not the header file only.
// This allows us access to static variables and globals (gasp!). This is more
// or less a close replica of the exposure that the functions being tested will
// get.
#include "neuron.cu"
#include "iteration.cu"

/* Study the spike pattern in a single neuron with a stepped input. 
 * The output of this program is meant to be viewed with the included spiking
 * visualizer in test/visualizer/. */
int test_single_neuron(void)
{
  int *dev_time;
  int host_time = 0;
  bool *dev_fired, host_fired[1];

  Neuron host_neurons[1];
  Neuron *dev_neurons;

  int host_rate[NEU_COUNT];
  int *dev_rate;

  bool host_fired_queue[NEU_COUNT * FIRED_RES], *dev_fired_queue;

  // Allocate memory on the GPU
  cudaMalloc( (void**)&dev_time, sizeof(int));
  cudaMalloc( (void**)&dev_neurons, sizeof(Neuron));
  cudaMalloc( (void**)&dev_rate, sizeof(int) * NEU_COUNT);
  cudaMalloc( (void**)&dev_fired_queue, sizeof(bool) * FIRED_RES * NEU_COUNT);

  // Initialization
  host_neurons[0].current = 0;
  fill_false(host_fired, 1);
  fill_false(host_fired_queue, NEU_COUNT * FIRED_RES);
  fill_zeros(host_rate, NEU_COUNT);
  cudaMemcpy(dev_time, &host_time, sizeof(int),
      cudaMemcpyHostToDevice);
  cudaMemcpy(dev_neurons, &host_neurons, sizeof(Neuron),
      cudaMemcpyHostToDevice);
  cudaMemcpy(dev_fired_queue, &host_fired_queue, sizeof(bool) * NEU_COUNT * FIRED_RES,
      cudaMemcpyHostToDevice);
  cudaMemcpy(dev_rate, &host_rate, sizeof(int) * NEU_COUNT,
      cudaMemcpyHostToDevice);

  while (host_time < 1000) {
    time_step<<<1,1>>>(dev_time);
    cudaMalloc( (void**)&dev_fired, sizeof(bool));
    cudaMemcpy(dev_fired, &host_fired, sizeof(bool), cudaMemcpyHostToDevice);

    if (host_time == 500) {
      // At t=400, give thalamic input of 4 to the neuron
      host_neurons[0].input = 13;
      cudaMemcpy(dev_neurons, host_neurons, sizeof(Neuron),
          cudaMemcpyHostToDevice);
    }

    if (host_time == 800) {
      // At t=800, remove all thalamic input
      host_neurons[0].input = 0.0f;
      cudaMemcpy(dev_neurons, host_neurons, sizeof(Neuron),
          cudaMemcpyHostToDevice);
    }

    find_firing_neurons<<<1,1>>>(dev_neurons, dev_time, dev_fired, dev_rate,
        dev_fired_queue, 1);
    update_current<<<1,1>>>(dev_neurons, NULL, dev_fired, 1);
    update_potential<<<1,1>>>(dev_neurons, NULL, 1);

    cudaMemcpy(host_neurons, dev_neurons, sizeof(Neuron),
        cudaMemcpyDeviceToHost);

    printf("[ %d, %10f],\n", host_time, host_neurons[0].potential);

    host_time++;
  }

  return 0;
}


int main(void) 
{
  test_single_neuron();

  return 0;
}
