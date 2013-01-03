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

Using the build/format program
---

```shell
tail -n+600 data/vib_table.data | head -n800 | ./build/format > a.js
```

Let's look at this part by part. The `tail` program will take the part of the
file after the first 600 lines. This will be passed on to the `head` program,
which will take the first 800 lines of the remaining lines. This 800 line unit
will now be passed to the build/format program which will accordingly format it
and write the output of the program to the a.js file.

[cuda-link]: http://developer.nvidia.com/cuda
