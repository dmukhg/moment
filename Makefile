LIBRARY_DIRS=-L /usr/lib64 -L /usr/local/cuda/lib64
LIBRARIES=-l cudart -l stdc++
INCLUDES=-I /usr/local/cuda/include -I /usr/local/cuda/cudart
COMPILER=--compiler-bindir /usr/bin/gcc-4.4

NVCC= nvcc
GCC=gcc
GCCFLAGS=-std=c99
NFLAGS= -arch=sm_12 $(INCLUDES) $(COMPILER) -m 64 $(LIBRARY_DIRS) $(LIBRARIES)
FFLAGS= -l cufft # For fourier modules
BUILD_DIR=build
TFLAGS= -Ilib

build/test-network-library: test/network-library.cu
	$(NVCC) $(NFLAGS) $(TFLAGS) -o $@ $^

# Application targets
#build/moment: main.o neuron.o iteration.o
#	$(NVCC) $(NFLAGS) -o $@ $^
#	rm $^
#
#main.o: lib/main.cu lib/main.cuh lib/neuron.cuh \
#				lib/iteration.cuh lib/defs.cuh
#	$(NVCC) $(NFLAGS) -c -o $@ $<
#
#neuron.o: lib/neuron.cu lib/neuron.cuh \
#				lib/defs.cuh
#	$(NVCC) $(NFLAGS) -c -o $@ $<
#
#iteration.o: lib/iteration.cu lib/iteration.cuh \
#				lib/defs.cuh
#	$(NVCC) $(NFLAGS) -c -o $@ $<
#
## Testing targets
#build/test-xor-benchmark: test/xor-benchmark.cu lib/neuron.cu* \
#									lib/iteration.cu* lib/defs.cuh
#	$(NVCC) $(NFLAGS) $(TFLAGS) -o $@ $<
#
#build/test-neuron: test/neuron-test.cu lib/neuron.cu* \
#									lib/iteration.cu* lib/defs.cuh
#	$(NVCC) $(NFLAGS) $(TFLAGS) -o $@ $< 
#
#build/test-2neurons: test/2neurons-test.cu lib/neuron.cu* \
#									lib/iteration.cu* lib/defs.cuh
#	$(NVCC) $(NFLAGS) $(TFLAGS) -o $@ $<
#
#build/test-vibration-single-neuron: test/vibration-single-neuron.cu lib/neuron.cu* \
#									lib/iteration.cu* lib/defs.cuh
#	$(NVCC) $(NFLAGS) $(TFLAGS) -o $@ $<
#
#build/test-frequency-to-neuron: test/frequency-to-neuron.cu lib/neuron.cu* \
#									lib/iteration.cu* lib/defs.cuh
#	$(NVCC) $(NFLAGS) $(TFLAGS) -o $@ $<
#
#build/format: test/format.cu
#	$(NVCC) $(NFLAGS) $(TFLAGS) -o $@ $<
#
#build/frequency-generator: test/frequency-generator.c
#	$(GCC) -lm $(GCCFLAGS) $(TFLAGS) -o $@ $<
#
#build/test-cluster: test/cluster.cu lib/network.cuh
#	$(NVCC) $(NFLAGS) $(TFLAGS) -o $@ $<
#
#build/fourier-transform: test/fourier.cu
#	$(NVCC) $(NFLAGS) $(TFLAGS) $(FFLAGS) -o $@ $<
#
.PHONY: clean 

clean:
	rm build/*
