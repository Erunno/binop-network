# Makefile for CUDA-based binary neural network implementation

# Directories
BIN_DIR := bin
OBJ_DIR := $(BIN_DIR)/obj

# Create directories if they don't exist
$(shell mkdir -p $(BIN_DIR))
$(shell mkdir -p $(OBJ_DIR))

# Compilers
CXX := g++
NVCC := nvcc

# Compiler flags
CXXFLAGS := -std=c++20 -O2 -Wall -Wextra
NVCCFLAGS := -std=c++20 -O2 -Wno-deprecated-gpu-targets --compiler-options -Wall,-Wextra --diag-suppress 326

# Include paths
INCLUDES := -I.

# Find all source files
CPP_SRCS := $(wildcard *.cpp)
CU_SRCS := $(wildcard *.cu)

# Object files
CPP_OBJS := $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(CPP_SRCS))
CU_OBJS := $(patsubst %.cu,$(OBJ_DIR)/%.o,$(CU_SRCS))
ALL_OBJS := $(CPP_OBJS) $(CU_OBJS)

# Main target
TARGET := $(BIN_DIR)/network

.PHONY: all clean

all: $(TARGET)

# Linking
$(TARGET): $(ALL_OBJS)
	@echo "Linking $@..."
	$(NVCC) $(NVCCFLAGS) $^ -o $@

# Compile C++ source files
$(OBJ_DIR)/%.o: %.cpp
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile CUDA source files
$(OBJ_DIR)/%.o: %.cu
	@echo "Compiling $<..."
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Clean rule
clean:
	@echo "Cleaning up..."
	rm -rf $(BIN_DIR)

# Dependencies
$(CPP_OBJS): | $(OBJ_DIR)
$(CU_OBJS): | $(OBJ_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)
