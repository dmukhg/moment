/* Provides a class to contain an SNN. */
#ifndef MOMENT_NETWORK_CUH

#define MOMENT_NETWORK_CUH

#include <stdlib.h>

#include "types.cuh"
#include "kernels.cuh"
#include "utils.cuh"

/* Class representing a network */
class Network {
  private: 
    Neuron *dev_neurons, *host_neurons;
    Connection *dev_connections, *host_connections;
    bool *dev_fired, *host_fired, *dev_fired_table, *host_fired_table;
    int *host_rate, *dev_rate, *dev_time, host_time, n_neurons, n_connections
      , fired_res;

  public:
    Network (int num_neurons, int num_connections) {
      n_neurons = num_neurons;
      n_connections = num_connections;

      fired_res = 100;

      host_time = 0;
      host_neurons = new Neuron[n_neurons];
      host_connections = new Connection[n_connections];
      host_rate = new int[n_neurons]; 
      host_fired = new bool[n_neurons];
      host_fired_table = new bool[n_neurons * fired_res];

      // Initialize
      fill_false(host_fired, n_neurons);
      fill_false(host_fired_table, n_neurons * fired_res);

      // Allocate
      cudaMalloc( (void**)&dev_time, sizeof(int) );
      cudaMalloc( (void**)&dev_neurons, sizeof(Neuron) * n_neurons);
      cudaMalloc( (void**)&dev_connections, sizeof(Connection) * n_connections);
      cudaMalloc( (void**)&dev_rate, sizeof(int) * n_neurons);
      cudaMalloc( (void**)&dev_fired, sizeof(bool) * n_neurons);
      cudaMalloc( (void**)&dev_fired_table, sizeof(bool) * n_neurons * fired_res);

      // Copy
      cudaMemcpy(dev_neurons, host_neurons, sizeof(Neuron) * n_neurons,
          cudaMemcpyHostToDevice);
      cudaMemcpy(dev_connections, host_connections, sizeof(Connection) * n_connections,
          cudaMemcpyHostToDevice);
      cudaMemcpy(dev_time, &host_time, sizeof(int),
          cudaMemcpyHostToDevice);
      cudaMemcpy(dev_rate, host_rate, sizeof(int) * n_neurons,
          cudaMemcpyHostToDevice);
      cudaMemcpy(dev_fired, host_fired, sizeof(bool) * n_neurons,
          cudaMemcpyHostToDevice);
      cudaMemcpy(dev_fired_table, host_fired_table, sizeof(bool) * n_neurons * fired_res,
          cudaMemcpyHostToDevice);
    };

    ~Network () {
      cudaFree(dev_neurons);
      cudaFree(dev_time);
      cudaFree(dev_connections);
      cudaFree(dev_rate);
      cudaFree(dev_fired);
      cudaFree(dev_fired_table);
    };
  

    // Accessors for time
    int time() {
      /* Returns the most recent value of host_time. This may or may not be in
       * sync with dev_time. */
      return time(false);
    };

    int time(bool sync) {
      /* Returns host_time. If sync is true, copies dev_time into host_time and
       * then returns it. */
      if (sync) {
        cudaMemcpy(&host_time, dev_time, sizeof(int),
            cudaMemcpyDeviceToHost);
      }

      return host_time;
    };

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
        cudaMemcpy(dev_connections, host_connections, 
            sizeof(Connection) * n_connections, cudaMemcpyHostToDevice);
      }
    };

    // Accessor for Spiking Rate
    int * spiking_rate() {
      return spiking_rate(false);
    };

    int * spiking_rate(bool sync) {
      /* Returns the spiking rate. If sync is true, copies from device and then
       * returns */
      if (sync) {
        cudaMemcpy(host_rate, dev_rate, sizeof(int) * n_neurons,
            cudaMemcpyDeviceToHost);
      }

      return host_rate;
    };


    // Kernel callers
    void time_step() {
      _time_step<<<1,1>>>(dev_time);
      host_time++;
    };

    void find_firing_neurons() {
      _find_firing_neurons<<<1,n_neurons>>>(dev_neurons, dev_time, dev_fired,
          dev_rate, dev_fired_table, n_neurons, fired_res);
    };

    void update_current() {
      _update_current<<<1,n_neurons>>>(dev_neurons, dev_connections, dev_fired,
          n_neurons);
    };

    void update_potential() {
      _update_potential<<<1,n_neurons>>>(dev_neurons, dev_connections,
          n_neurons);

      // Reset fired to false
     cudaMemcpy(dev_fired, host_fired, sizeof(bool) * n_neurons,
        cudaMemcpyHostToDevice);
    };


    // Build Connections
    /* Build connections assuming complete connectivity. */
    void build_connections(int num_input, int num_hidden, int num_output) {
      if (num_input + num_hidden + num_output != n_neurons) {
        // Ensure the parameters supplied are indeed adding up to the number of
        // neurons
        exit(1);
      }

      if (num_input*num_hidden + num_hidden*num_output != n_connections) {
        // Ensure the number of connections add up
        exit(1);
      }

      int i, j, num_i_and_h, num_c_so_far;

      // Fill up the connections between the input layer and the hidden layer
      for (i=0; i<num_input; i++) {
        host_neurons[i].connection = num_hidden * i; 

        for (j=0; j<num_hidden; j++) {
          host_connections[i*num_hidden + j].neuron = num_input + j;
          host_connections[i*num_hidden + j].weight = 20.0;

          if (j == num_hidden - 1) {
            host_connections[i*num_hidden + j].next = -1;
          } else {
            host_connections[i*num_hidden + j].next = num_hidden * i + j + 1;
          }
        }
      }

      // Fill up the connections between the hidden layer and the output layer
      num_i_and_h = num_input + num_hidden;
      num_c_so_far = num_input * num_hidden;

      for (i=0; i<num_hidden; i++) {
        host_neurons[num_input + i].connection = num_c_so_far + num_output * i;

        for (j=0; j<num_output; j++) {
          host_connections[num_c_so_far + i*num_output + j].neuron = num_i_and_h + j;
          host_connections[num_c_so_far + i*num_output + j].weight = 20.0;

          if (j == num_output - 1) {
            host_connections[num_c_so_far + i*num_output + j].next = -1;
          } else {
            host_connections[num_c_so_far + i*num_output + j].next = num_c_so_far | num_hidden * i + j + 1;
          }
        }
      }

      // Copy to GPU
      cudaMemcpy(dev_neurons, host_neurons,
          sizeof(Neuron) * n_neurons, cudaMemcpyHostToDevice);
      cudaMemcpy(dev_connections, host_connections, 
          sizeof(Connection) * n_connections, cudaMemcpyHostToDevice);
    };
};

#endif
