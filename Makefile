NVCC= nvcc
BUILD_DIR=build
TFLAGS= -Isrc

# Application targets
build/moment: main.o neuron.o iteration.o
	$(NVCC) -o $@ $^
	rm $^

main.o: src/main.cu src/main.cuh src/neuron.cuh \
				src/iteration.cuh src/defs.cuh
	$(NVCC) -c -o $@ $<

neuron.o: src/neuron.cu src/neuron.cuh \
				src/defs.cuh
	$(NVCC) -c -o $@ $<

iteration.o: src/iteration.cu src/iteration.cuh \
				src/defs.cuh
	$(NVCC) -c -o $@ $<

# Testing targets
test: build/test-neuron build/test-network
	./build/test-neuron
	./build/test-network

build/test-neuron: test/neuron-test.cu src/neuron.cu* \
									src/iteration.cu* src/defs.cuh
	$(NVCC) $(TFLAGS) -o $@ $< 

build/test-network: test/network-test.cu src/neuron.cu* \
									src/iteration.cu* src/defs.cuh
	$(NVCC) $(TFLAGS) -o $@ $<

.PHONY: clean 

clean:
	rm build/*
	rm *.o
