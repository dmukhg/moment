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

tests: test/build/network-library test/build/kernels

test/build/network-library: test/network-library.cu lib/network.cuh \
						lib/kernels.cuh lib/types.cuh lib/defs.cuh
	$(NVCC) $(NFLAGS) $(TFLAGS) -o $@ $<

test/build/kernels: test/kernels.cu lib/network.cuh \
						lib/kernels.cuh lib/types.cuh lib/defs.cuh
	$(NVCC) $(NFLAGS) $(TFLAGS) -o $@ $<


tools: build/frequency-generator build/fourier-transform \
	build/format

build/format: src/format.cu
	$(NVCC) $(NFLAGS) $(TFLAGS) -o $@ $<

build/frequency-generator: src/frequency-generator.c
	$(GCC) -lm $(GCCFLAGS) $(TFLAGS) -o $@ $<

build/fourier-transform: src/fourier.cu
	$(NVCC) $(NFLAGS) $(TFLAGS) $(FFLAGS) -o $@ $<

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

.PHONY: clean 

clean:
	rm build/*
	rm test/build/*
