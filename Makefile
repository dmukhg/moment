NVCC= nvcc
BUILD_DIR=build

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
test: test-neuron

.PHONY: clean 

clean:
	rm build/*
	rm *.o

