#include "network.hpp"
#include <stdio.h>
#include <cuda_runtime.h>

namespace netw {


// Implement the compute function for the neuron class with host and device support
template <typename type_config>
__host__ __device__ __forceinline__ 
typename type_config::value_type neuron<type_config>::compute(const typename type_config::value_type* input_layer) const {
    switch(op) {
        case 0: // AND
            return input_layer[input1] & input_layer[input2];
        case 1: // OR
            return input_layer[input1] | input_layer[input2];
        case 2: // XOR
            return input_layer[input1] ^ input_layer[input2];
        case 3: // NOT
            return ~input_layer[input1];
        case 4: // NoOP
            return input_layer[input1];
        default:
            printf("Invalid operation code: %d\n", op);
            return 0;
    }
}

template <typename type_config>
__host__ __device__ __forceinline__
void layer<type_config>::compute(const typename type_config::value_type* input, typename type_config::value_type* output) {
    for (size_type i = 0; i < size; ++i) {
        output[i] = neurons[i].compute(input);
    }
}

// Implement the compute function for the network class
template <typename type_config>
__host__ __device__ __forceinline__
void network<type_config>::compute(const typename type_config::value_type* input, typename type_config::value_type* output) {
    value_type* current_input = const_cast<value_type*>(input);
    value_type* current_output = working_memory_1;
    
    for (size_type i = 0; i < layer_count - 1; ++i) {
        layers[i].compute(current_input, current_output);

        auto temp = current_input;
        current_input = current_output;
        current_output = temp;
    }
    
    layers[layer_count - 1].compute(current_input, output);
}

// Explicit instantiation for the types we'll be using
template __host__ __device__ __forceinline__ 
SMALLEST_TYPES::value_type neuron<SMALLEST_TYPES>::compute(const SMALLEST_TYPES::value_type* input_layer) const;

template __host__ __device__ __forceinline__
void layer<SMALLEST_TYPES>::compute(const SMALLEST_TYPES::value_type* input, SMALLEST_TYPES::value_type* output);

template __host__ __device__ __forceinline__
void network<SMALLEST_TYPES>::compute(const SMALLEST_TYPES::value_type* input, SMALLEST_TYPES::value_type* output);

} // namespace netw

