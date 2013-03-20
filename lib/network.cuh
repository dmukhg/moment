/* Provides a struct to contain an SNN. */
#ifndef MOMENT_NETWORK_CUH

#define MOMENT_NEURON_CUH

#include <stdlib.h>

/* Structure representing a connection. Has the id of the post
 * synaptic neuron, the synaptic weight and the index of the next
 * connection.
 */
struct Connection {
  int neuron;
  float weight;
  int next;
};

/* Structure representing a neuron in the Izhikewich model.
 *
 * current represents Thalamic Input.
 * connection represents the first connection that this neuron has.
 */
struct Neuron {
  float potential // 'v' Membrane potential
      , recovery  // 'u' Negative feedback 
      , input;    // In case this is an input neuron

  int current; // 'I' Input current. Sum of all input potential


  unsigned int connection;

  Neuron() : current(0), potential(0.0), recovery(0.0), connection(0), input(0.0) {}
};

__global__ void _time_step(int *dev_time) {
  /* Increment the time value in the device by 1 */
  *dev_time += 1;
};

/* Class representing a network */
class Network {
  private: 
    Neuron *dev_neurons, *host_neurons;
    Connection *dev_connections, *host_connections;
    bool *dev_fired, host_fired;
    int *dev_time, host_time, n_neurons, n_connections;

  public:
    Network (int num_neurons, int num_connections) {
      n_neurons = num_neurons;
      n_connections = num_connections;

      host_time = 0;
      host_neurons = (Neuron*)malloc(sizeof(Neuron) * num_neurons);
      host_connections = new Connection[num_connections];

      // Allocate
      cudaMalloc( (void**)&dev_time, sizeof(int) );
      cudaMalloc( (void**)&dev_neurons, sizeof(Neuron) * num_neurons);
      cudaMalloc( (void**)&dev_connections, sizeof(Connection) * num_connections);

      // Copy
      cudaMemcpy(dev_time, &host_time, sizeof(int),
          cudaMemcpyHostToDevice);
    };
  

    // Accessors for time
    int time() {
      /* Returns the most recent value of host_time. This may or may not be in
       * sync with dev_time. */
      return time(false);
    }

    int time(bool sync) {
      /* Returns host_time. If sync is true, copies dev_time into host_time and
       * then returns it. */
      if (sync) {
        cudaMemcpy(&host_time, dev_time, sizeof(int),
            cudaMemcpyDeviceToHost);
      }

      return host_time;
    }

    void time(int t) {
      time(t, false);
    }

    void time(int t, bool sync) {
      host_time = t;

      if (sync) {
        cudaMemcpy(dev_time, &host_time, sizeof(int),
          cudaMemcpyHostToDevice);
      }
    }

    // Accessors for Neurons
    Neuron * neurons() {
      /* Returns the most recent copy of host_neurons. This may or may not be
       * in sync with dev_neurons. */
      return neurons(false);
    };

    Neuron * neurons(bool sync) {
      /* Returns host_neurons. If sync is true, copies dev_neurons and then
       * returns host_neurons. */
      if (sync) {
        cudaMemcpy(host_neurons, dev_neurons, sizeof(Neuron) * n_neurons,
            cudaMemcpyDeviceToHost);
      }

      return host_neurons;
    };

    void neurons(Neuron * host_neuron) {
      neurons(host_neurons, false);
    };

    void neurons(Neuron * value, bool sync) {
      /* Sets this.host_neurons to supplied argument. If sync is true, copies
       * host_neurons to dev_neurons */
      host_neurons = value;

      if (sync) {
        cudaMemcpy(dev_neurons, host_neurons, sizeof(Neuron) * n_neurons,
            cudaMemcpyHostToDevice);
      }
    };

    // Accessors for Connections
    Connection * connections() {
      /* Returns the most recent copy of host_connections. This may or may not
       * be in sync with dev_connections. */
      return connections(false);
    };

    Connection * connections(bool sync) {
      /* Returns host_connections. If sync is true, copies dev_connections and
       * then returns host_connections. */
      if (sync) {
        cudaMemcpy(host_connections, dev_connections, sizeof(Connection) *
            n_connections, cudaMemcpyDeviceToHost);
      }

      return host_connections;
    };

    void connections(Connection * host_connections) {
      connections(host_connections, false);
    };

    void connections(Connection * value, bool sync) {
      /* Sets this.host_connections to supplied argument. If sync is true,
       * copies host_connections to dev_connections*/
      host_connections = value;

      if (sync) {
        cudaMemcpy(dev_connections, host_connections, sizeof(Connection) * n_connections,
            cudaMemcpyHostToDevice);
      }
    };

    void time_step() {
      _time_step<<<1,1>>>(dev_time);
      host_time++;
    };
};

#endif
