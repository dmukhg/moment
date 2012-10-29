#include <stdio.h>

#include "main.cuh"

/* Step forward in time. */
__global__ void time_step(int *dev_tim) 
{
    *dev_time += 1;
}


int main( void ) 
{
    int *dev_time;
    int host_time = 0;
    time_t lt;

    // Allocate memory on the GPU
    cudaMalloc( (void**)&dev_time, sizeof(int) );

    // Copy the time to the GPU
    cudaMemcpy(dev_time, &host_time, sizeof(int),
        cudaMemcpyHostToDevice);

    // XXX The limit shouldn't be iterations. Discuss!
    while (host_time < ITERATIONS) {
        // Step forward in time.  Since this is a part of the global memory,
        // you only need to do it via one thread.
        time_step<<<1,1>>>(dev_time);

        // Copy the time back to the CPU
        cudaMemcpy(&host_time, dev_time, sizeof(int),
            cudaMemcpyDeviceToHost);
    }

    // Free memory on GPU
    cudaFree(dev_time);

    return 0;
}
