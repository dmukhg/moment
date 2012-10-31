#include "utils.cuh"

/* Given a neuron index and a connection index, convert to an offset
 * useable on the 1D array that is connections.
 */
int get_index(int neuron, int connection) {
  return neuron*PSYNCONN + connection;
}

/* GPU version of the same */
__device__ int dev_get_index(int neuron, int connection) {
  return neuron*PSYNCONN + connection;
}
