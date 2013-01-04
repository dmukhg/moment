moment
===

Moment is a spiking neural network simulator that works on the
[CUDA][cuda-link] platform.  Rather than use pthreads or equivalents
to parallelize the neurons, moment relies on the highly scalable and
parallel cuda architecture.

A question that might frequently arise is why choose to parallelize such an
application? The answer lies in the final application of such a network. If a
neural network is to be put to on-line use, the output should be generated as
quickly as possible, as close to real-time as possible. Parallelization is
indispensable for this kind of application.

Let's see how this project proceeds.

License
---

Moment is licensed under the MIT License.  Please have a look at the
included LICENSE file.

Using the test/visualizer program
---

Moment includes a spike pattern viewer. It is written in Javascript and uses
[jQuery][jq] and [Flot][flot]. First, either open the
test/visualizer/index.html file directly in a browser or start a webserver by
using something like

```shell
python -m SimpleHTTPServer
```

while in the root directory. Once this server is running, navigate to
`localhost:8000/test/visualizer` to see the interface. Copy _all_ the output of
any of the test programs and paste it between the two square brackets in the
interface. Hit draw to form a plot.

There are 2 plotting areas provided so that you can compare two spiketrains.

Using the build/format program
---

The `build/format` program takes a file or a stream with one floating point
number on each line, and converts that into a format that the Spiking
Visualizer can easily use.

```shell
tail -n+600 data/vib_table.data | head -n800 | ./build/format > a.js
```

Let's look at this part by part. The `tail` program will take the part of the
file after the first 600 lines. This will be passed on to the `head` program,
which will take the first 800 lines of the remaining lines. This 800 line unit
will now be passed to the build/format program which will accordingly format it
and write the output of the program to the a.js file.

[cuda-link]: http://developer.nvidia.com/cuda
[jq]: http://jquery.com/
[flot]: http://github.com/flot/
