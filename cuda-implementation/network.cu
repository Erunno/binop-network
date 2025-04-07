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

template <typename type_config>
__host__ __device__ __forceinline__
void network<type_config>::compute(const typename type_config::value_type* input, typename type_config::value_type* output) {
    value_type* current_input = working_memory_1;
    value_type* current_output = working_memory_2;
    
    layers[0].compute(input, current_input);

    for (size_type i = 1; i < layer_count - 1; ++i) {
        layers[i].compute(current_input, current_output);

        auto temp = current_input;
        current_input = current_output;
        current_output = temp;
    }
    
    layers[layer_count - 1].compute(current_input, output);
}

template <typename type_config>
__host__ __device__ __forceinline__
void neuron_change_list<type_config>::apply_changes(network<type_config>* net) {
    for (size_type i = 0; i < count; ++i) {
        const neuron_change<type_config>& change = changes[i];
        
        if (change.layer_index < net->layer_count) {
            layer<type_config>& layer = net->layers[change.layer_index];
            
            if (change.neuron_index < layer.size) {
                neuron<type_config>& target_neuron = layer.neurons[change.neuron_index];
                
                target_neuron.op = change.op;
                target_neuron.input1 = change.input1;
                target_neuron.input2 = change.input2;
            }
        }
    }
}

template <typename type_config>
__host__ __device__ __forceinline__
void layer<type_config>::compute_with_changed_neurons(
    neuron_change_list<type_config> changes, size_type layer_index,
    const typename type_config::value_type* input, typename type_config::value_type* output) {
    
    for (size_type i = 0; i < size; ++i) {
        bool is_modified = false;
        neuron<type_config> modified_neuron;
        
        for (size_type c_idx = 0; c_idx < changes.count; c_idx++) {
            if (changes.changes[c_idx].layer_index == layer_index && 
                changes.changes[c_idx].neuron_index == i) {

                // Found a change for this neuron
                modified_neuron.op = changes.changes[c_idx].op;
                modified_neuron.input1 = changes.changes[c_idx].input1;
                modified_neuron.input2 = changes.changes[c_idx].input2;
                is_modified = true;
                break;
            }
        }
        
        if (is_modified) {
            output[i] = modified_neuron.compute(input);
        } else {
            output[i] = neurons[i].compute(input);
        }
    }
}

template <typename type_config>
__host__ __device__ __forceinline__
void network<type_config>::compute_with_changed_neurons(
    neuron_change_list<type_config> changes,
    const value_type* input, value_type* output) {

    value_type* current_input = working_memory_1;
    value_type* current_output = working_memory_2;
    
    layers[0].compute_with_changed_neurons(changes, 0, input, current_input);
    
    for (size_type i = 1; i < layer_count - 1; ++i) {
        layers[i].compute_with_changed_neurons(changes, i, current_input, current_output);
        
        // Swap buffers for next layer
        value_type* temp = current_input;
        current_input = current_output;
        current_output = temp;
    }
    
    layers[layer_count - 1].compute_with_changed_neurons(
        changes, layer_count - 1, current_input, output);
}

// Explicit instantiation for the types we'll be using
template __host__ __device__ __forceinline__ 
SMALLEST_TYPES::value_type neuron<SMALLEST_TYPES>::compute(const SMALLEST_TYPES::value_type* input_layer) const;

template __host__ __device__ __forceinline__
void layer<SMALLEST_TYPES>::compute(const SMALLEST_TYPES::value_type* input, SMALLEST_TYPES::value_type* output);

template __host__ __device__ __forceinline__
void network<SMALLEST_TYPES>::compute(const SMALLEST_TYPES::value_type* input, SMALLEST_TYPES::value_type* output);

template __host__ __device__ __forceinline__
void neuron_change_list<SMALLEST_TYPES>::apply_changes(network<SMALLEST_TYPES>* net);

template __host__ __device__ __forceinline__
void layer<SMALLEST_TYPES>::compute_with_changed_neurons(
    neuron_change_list<SMALLEST_TYPES> changes, SMALLEST_TYPES::size_type layer_index,
    const SMALLEST_TYPES::value_type* input, SMALLEST_TYPES::value_type* output);

template __host__ __device__ __forceinline__
void network<SMALLEST_TYPES>::compute_with_changed_neurons(
    neuron_change_list<SMALLEST_TYPES> changes,
    const SMALLEST_TYPES::value_type* input, SMALLEST_TYPES::value_type* output);

} // namespace netw

