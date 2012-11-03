#include "iteration.cuh"

/* Increment the time-step by 1.  
 * For use only with a single block and a single thread */
__global__ void time_step(int *dev_time)
{
    *dev_time += 1;
}
