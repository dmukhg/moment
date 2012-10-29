NVCC= nvcc
BUILD_DIR=build

main: src/main.cuh src/main.cu
	$(NVCC) -o $(BUILD_DIR)/moment src/main.cu
