NVCC= nvcc
BUILD_DIR=build

moment: main.o connections.o utils.o src/defs.cuh
	$(NVCC) -o $(BUILD_DIR)/moment main.o utils.o
	rm main.o connections.o utils.o

main.o: src/main.cuh src/main.cu src/connections.cuh \
				src/utils.cuh src/defs.cuh
	$(NVCC) -c -o main.o src/main.cu

connections.o: src/connections.cuh src/connections.cu \
				src/defs.cuh
	$(NVCC) -c -o connections.o src/connections.cu

utils.o: src/utils.cuh src/utils.cu src/defs.cuh
	$(NVCC) -c -o utils.o src/utils.cu
