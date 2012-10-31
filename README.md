moment
===

Moment is a spiking neural network simulator that works on the
[CUDA][cuda-link] platform.  Rather than use pthreads or equivalents
to parallelize the neurons, moment relies on the highly scalable and
parallel cuda architecture.

Spiking neural networks use temporal coding and as such, require heavy
parallelization since any idle time can lead to loss of information
which is unacceptable.

Let's see how this project proceeds.

License
---

Moment is licensed under the MIT License.  Please have a look at the
included LICENSE file.

[cuda-link]: http://developer.nvidia.com/cuda
