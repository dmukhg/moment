LIBRARY_DIRS=-L /usr/lib64 -L /usr/local/cuda/lib64
LIBRARIES=-l cudart -l stdc++
INCLUDES=-I /usr/local/cuda/include -I /usr/local/cuda/cudart
COMPILER=--compiler-bindir /usr/bin/gcc-4.4

NVCC= nvcc
NFLAGS= -arch=sm_12 $(INCLUDES) $(COMPILER) -m 64 $(LIBRARY_DIRS) $(LIBRARIES)
BUILD_DIR=build
TFLAGS= -Isrc

# Application targets
build/moment: main.o neuron.o iteration.o
	$(NVCC) $(NFLAGS) -o $@ $^
	rm $^

main.o: src/main.cu src/main.cuh src/neuron.cuh \
				src/iteration.cuh src/defs.cuh
	$(NVCC) $(NFLAGS) -c -o $@ $<

neuron.o: src/neuron.cu src/neuron.cuh \
				src/defs.cuh
	$(NVCC) $(NFLAGS) -c -o $@ $<

iteration.o: src/iteration.cu src/iteration.cuh \
				src/defs.cuh
	$(NVCC) $(NFLAGS) -c -o $@ $<

# Testing targets
build/test-xor-benchmark: test/xor-benchmark.cu src/neuron.cu* \
									src/iteration.cu* src/defs.cuh
	$(NVCC) $(NFLAGS) $(TFLAGS) -o $@ $<

build/test-neuron: test/neuron-test.cu src/neuron.cu* \
									src/iteration.cu* src/defs.cuh
	$(NVCC) $(NFLAGS) $(TFLAGS) -o $@ $< 

build/test-2neurons: test/2neurons-test.cu src/neuron.cu* \
									src/iteration.cu* src/defs.cuh
	$(NVCC) $(NFLAGS) $(TFLAGS) -o $@ $<

build/test-vibration-single-neuron: test/vibration-single-neuron.cu src/neuron.cu* \
									src/iteration.cu* src/defs.cuh
	$(NVCC) $(NFLAGS) $(TFLAGS) -o $@ $<

build/test-frequency-to-neuron: test/frequency-to-neuron.cu src/neuron.cu* \
									src/iteration.cu* src/defs.cuh
	$(NVCC) $(NFLAGS) $(TFLAGS) -o $@ $<

build/format: test/format.cu
	$(NVCC) -o $@ $<

build/frequency-generator: test/frequency-generator.cu
	$(NVCC) -o $@ $<

.PHONY: clean 

clean:
	rm build/*
	rm *.o
