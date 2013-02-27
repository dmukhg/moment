/* Test network functionality */
#define B2S(c) ((c) ? 't': 'f') 
#define NEU_COUNT 2

#include <stdio.h>

// Note that we are including the source file and not the header file only.
// This allows us access to static variables and globals (gasp!). This is more
// or less a close replica of the exposure that the functions being tested will
// get.
#include "neuron.cu"
#include "iteration.cu"

int test_two_neuron_network(void)
{
  int *dev_time;
  int host_time = 0;

  Neuron host_neurons[2];
  Neuron *dev_neurons;

  int host_rate[NEU_COUNT];
  int *dev_rate;

  Connection host_connections[1];
  Connection *dev_connections;

  bool *dev_fired, host_fired[2];
  bool host_fired_queue[NEU_COUNT * FIRED_RES], *dev_fired_queue;

  // Allocate memory on the GPU
  cudaMalloc( (void**)&dev_time, sizeof(int));
  cudaMalloc( (void**)&dev_neurons, sizeof(Neuron)*2);
  cudaMalloc( (void**)&dev_connections, sizeof(Connection));
  cudaMalloc( (void**)&dev_rate, sizeof(int) * NEU_COUNT);
  cudaMalloc( (void**)&dev_fired_queue, sizeof(bool) * FIRED_RES * NEU_COUNT);

  // Initialization
  fill_false(host_fired, 2);
  fill_false(host_fired_queue, NEU_COUNT * FIRED_RES);
  fill_zeros(host_rate, NEU_COUNT);
  host_neurons[0].connection = 0;
  host_connections[0].next = -1;
  host_connections[0].weight = 20.0;
  host_connections[0].neuron = 1; 

  // Copy all to device
  cudaMemcpy(dev_time, &host_time, sizeof(int),
      cudaMemcpyHostToDevice);
  cudaMemcpy(dev_neurons, &host_neurons, sizeof(Neuron)*2,
      cudaMemcpyHostToDevice);
  cudaMemcpy(dev_connections, &host_connections, sizeof(Connection),
      cudaMemcpyHostToDevice);
  cudaMemcpy(dev_fired_queue, &host_fired_queue, sizeof(bool) * NEU_COUNT * FIRED_RES,
      cudaMemcpyHostToDevice);
  cudaMemcpy(dev_rate, &host_rate, sizeof(int) * NEU_COUNT,
      cudaMemcpyHostToDevice);


  while (host_time < 1000) {
    time_step<<<1,1>>>(dev_time);
    cudaMalloc( (void**)&dev_fired, sizeof(bool)*2);
    cudaMemcpy(dev_fired, &host_fired, sizeof(bool)*2, cudaMemcpyHostToDevice);

    if (host_time == 500) {
      // At t=500, give thalamic input of 4 to the neuron
      host_neurons[0].input = 4;
      cudaMemcpy(dev_neurons, host_neurons, sizeof(Neuron)*2,
          cudaMemcpyHostToDevice);
    }

    find_firing_neurons<<<1,NEU_COUNT>>>(dev_neurons, dev_time, dev_fired, dev_rate,
        dev_fired_queue, NEU_COUNT);
    update_current<<<1,2>>>(dev_neurons, dev_connections, dev_fired, 2);
    update_potential<<<1,2>>>(dev_neurons, dev_connections, 2);

    cudaMemcpy(host_neurons, dev_neurons, sizeof(Neuron)*2,
        cudaMemcpyDeviceToHost);

    //printf("[ %d, %10d],\n", host_time, host_neurons[1].current);
    printf("[ %d, %10f],\n", host_time, host_neurons[1].potential);

    cudaFree(&dev_fired);
    host_time++;
  }

  return 0;
}

int main() {
  test_two_neuron_network();

  return 0;
}
