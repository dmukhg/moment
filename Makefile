NVCC= nvcc
NFLAGS= -arch=sm_12
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
build/test-neuron: test/neuron-test.cu src/neuron.cu* \
									src/iteration.cu* src/defs.cuh
	$(NVCC) $(NFLAGS) $(TFLAGS) -o $@ $< 

build/test-2neurons: test/2neurons-test.cu src/neuron.cu* \
									src/iteration.cu* src/defs.cuh
	$(NVCC) $(NFLAGS) $(TFLAGS) -o $@ $<

build/test-vibration-single-neuron: test/vibration-single-neuron.cu src/neuron.cu* \
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
