#include <stdio.h>

#include "defs.cuh"
#include "utils.cuh"
#include "connections.cuh"
#include "main.cuh"

/* Increment the time-step by 1.  

 * Note: Use only with a single block * and a single thread */
__global__ void time_step(int *dev_time)
{
    *dev_time += 1;
}


/* The main moment loop */
int main( void ) 
{
    int *dev_time;
    int host_time = 0;

    Connection host_connections[NUMNEURON * PSYNCONN]; 
    Connection *dev_connections;

    // Allocate memory on the GPU
    cudaMalloc( (void**)&dev_time, sizeof(int) );
    cudaMalloc( (void**)&dev_connections,
        sizeof(Connection) * NUMNEURON * PSYNCONN);

    // Copy the time to the GPU
    cudaMemcpy(dev_time, &host_time, sizeof(int),
        cudaMemcpyHostToDevice);

    // Copy the connections to the GPU
    cudaMemcpy(dev_connections, &host_connections,
        sizeof(Connection) * NUMNEURON * PSYNCONN,
        cudaMemcpyHostToDevice);

    while (host_time < ITERATIONS) {
        // Step forward in time.  Since this is a part of the global
        // memory, you only need to do it via one thread.
        time_step<<<1,1>>>(dev_time);

        // copy the time back to the cpu
        cudaMemcpy(&host_time, dev_time, sizeof(int),
            cudaMemcpyDeviceToHost); 
    }

    cudaMemcpy(&host_connections, dev_connections,
        sizeof(Connection) * NUMNEURON * PSYNCONN,
        cudaMemcpyDeviceToHost);

    // Free memory on GPU
    cudaFree(dev_time);

    return 0; 
}
