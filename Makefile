SOURCE_DIR = src
BUILD_DIR = build
EXECUTABLE = eigensolver

CXX = g++
CXXFLAGS = -m64 -O3 -Wall -std=c++11

NVCC = nvcc
NVCCFLAGS = -m64 -O3 -arch compute_20 -std=c++11

LDFLAGS = -L/usr/local/cuda/lib64/ -lcudart -lcusparse

INCLUDES = $(addprefix $(SOURCE_DIR)/, matrix.h graph_io.h linear_algebra.h cuda_algebra.h cycle_timer.h)
OBJECTS = $(addprefix $(BUILD_DIR)/, main.o cuda_algebra.o)

.PHONY: all clean

all: $(EXECUTABLE)

$(BUILD_DIR):
	mkdir -p $@

$(OBJECTS): $(INCLUDES) | $(BUILD_DIR)

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SOURCE_DIR)/%.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/%.o: $(SOURCE_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

clean:
	rm -rf $(BUILD_DIR) $(EXECUTABLE)
