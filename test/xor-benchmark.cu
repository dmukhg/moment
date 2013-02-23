/* Implement a learning method and try the xor benchmark */

#define NEU_COUNT 8
#define CON_COUNT 13

#include <stdio.h>

#include "neuron.cu"
#include "iteration.cu"

int xor_benchmark(void)
{
  int *dev_time;
  int host_time = 0;

  Neuron host_neurons[NEU_COUNT];
  Neuron *dev_neurons;

  Connection host_connections[CON_COUNT];
  Connection *dev_connections;

  bool *dev_fired, host_fired[NEU_COUNT];

  // Allocate memory on the GPU
  cudaMalloc( (void**)&dev_time, sizeof(int));
  cudaMalloc( (void**)&dev_neurons, sizeof(Neuron) * NEU_COUNT);
  cudaMalloc( (void**)&dev_connections, sizeof(Connection) * CON_COUNT);

  // Initialization
  fill_false(host_fired, NEU_COUNT);
  
  // Create the network architecture 
  host_neurons[0].connection = 0; // From neuron 0
  host_connections[0].neuron = 3;
  host_connections[0].next = 1;

  host_connections[1].neuron = 4;
  host_connections[1].next = 2;
  
  host_connections[2].neuron = 5;
  host_connections[2].next = -1;

  host_neurons[1].connection = 3; // From neuron 1
  host_connections[3].neuron = 3;
  host_connections[0].next = 4;

  host_connections[4].neuron = 4;
  host_connections[4].next = 5;

  host_connections[5].neuron = 5;
  host_connections[5].next = -1;

  host_neurons[2].connection = 6; // From neuron 2
  host_connections[6].neuron = 3;
  host_connections[6].next = 7;

  host_connections[7].neuron = 4;
  host_connections[7].next = 8;
  
  host_connections[8].neuron = 5;
  host_connections[8].next = -1;

  host_neurons[3].connection = 9;
  host_connections[9].neuron = 7;
  host_connections[9].next = -1;

  host_neurons[4].connection = 10;
  host_connections[10].neuron = 7;
  host_connections[10].next = -1;

  host_neurons[5].connection = 11;
  host_connections[11].neuron = 7;
  host_connections[11].next = -1;

  host_neurons[6].connection = 12;
  host_connections[12].neuron = 7;
  host_connections[12].next = -1;

  for (int i=0; i<13; i++) {
    host_connections[i].weight = 10.0;
  }

  // Copy all to device
  cudaMemcpy(dev_time, &host_time, sizeof(int),
      cudaMemcpyHostToDevice);
  cudaMemcpy(dev_neurons, &host_neurons, sizeof(Neuron) * NEU_COUNT,
      cudaMemcpyHostToDevice);
  cudaMemcpy(dev_connections, &host_connections, sizeof(Connection) * CON_COUNT,
      cudaMemcpyHostToDevice);

  while (host_time < 1000) {
    time_step<<<1,1>>>(dev_time);

    cudaMalloc( (void**)&dev_fired, sizeof(bool) * NEU_COUNT);
    cudaMemcpy(dev_fired, &host_fired, sizeof(bool) * NEU_COUNT,
      cudaMemcpyHostToDevice);

    if (host_time == 500) {
      host_neurons[1].input = 14;
      cudaMemcpy(dev_neurons, &host_neurons, sizeof(Neuron) * NEU_COUNT,
          cudaMemcpyHostToDevice);
    }

    find_firing_neurons<<<1,NEU_COUNT>>>(dev_neurons, dev_fired, NEU_COUNT);
    update_current<<<1,NEU_COUNT>>>(dev_neurons, dev_connections, dev_fired, 
        NEU_COUNT);
    update_potential<<<1,NEU_COUNT>>>(dev_neurons, dev_connections, NEU_COUNT);

    cudaMemcpy(host_neurons, dev_neurons, sizeof(Neuron) * NEU_COUNT,
        cudaMemcpyDeviceToHost);


    printf("[ %d, %10f],\n", host_time, host_neurons[7].potential);

    cudaFree(&dev_fired);
    host_time++;
  }

  return 0;
}


int main() {
  xor_benchmark();

  return 0;
}
