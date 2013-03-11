#define NX 256
#define BATCH 10

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include <cufft.h>

int test_fourier_transform(int count) {
  char line[LINE_MAX];
  int time;
  float host_data[count];
  float tmp;

  // Cuda specifics
  float *dev_data;
  cufftComplex *dev_signal, host_signal[count/2];

  cudaMalloc((void**)&dev_data, count * sizeof(float));
  cudaMalloc((void**)&dev_signal, count/2*sizeof(cufftComplex));

  for (time = 0; time < count; time++) {
    fgets(line, LINE_MAX, stdin);
    host_data[time] = strtof(line, NULL);
  }

  // Copy all to device
  cudaMemcpy(dev_data, &host_data, sizeof(float) * count,
      cudaMemcpyHostToDevice);

  // CUFFT plan
  cufftHandle planF;
  cufftPlan1d(&planF, count, CUFFT_R2C, 1);

  // Tranform signal
  cufftExecR2C(planF, dev_data, dev_signal);

  // Get back the device data
  cudaMemcpy(host_signal, dev_signal, sizeof(cufftComplex) * count/2,
      cudaMemcpyDeviceToHost);

  // Print data
  for (int i=0; i<count/2; i++) {
    tmp = host_signal[i].x * host_signal[i].x + host_signal[i].y * host_signal[i].y;
    printf("[%d, %10f],\n", i, tmp);
  }

  return 0;
}

int main(int argc, char *argv[]) {
  if (argc == 1) {
    printf("Usage: cat data | fourier-transform <count>");
    return 0;
  }

  int count = strtol(argv[1], NULL, 10);

  return test_fourier_transform(count);
}
