#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <cstddef>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

namespace netw {

template <typename type_config>
struct op {
    using index_type = typename type_config::index_type;

    static constexpr index_type AND = 0; 
    static constexpr index_type OR = 1;
    static constexpr index_type XOR = 2;
    static constexpr index_type NOT = 3;
    static constexpr index_type NoOP = 4;
};

template <typename type_config>
struct neuron {
    using index_type = typename type_config::index_type;
    using value_type = typename type_config::value_type;

    index_type op;
    index_type input1;
    index_type input2;

    __host__ __device__ value_type compute(const value_type* input_layer) const;
};

template <typename type_config>
struct layer {
    using size_type = typename type_config::size_type;
    using value_type = typename type_config::value_type;

    size_type size;
    size_type input_size;
    neuron<type_config>* neurons;

    __host__ __device__ void compute(const value_type* input, value_type* output);
};

template <typename type_config>
struct network {
    using size_type = typename type_config::size_type;
    using value_type = typename type_config::value_type;

    size_type input_size;
    size_type layer_count;
    layer<type_config>* layers;
    neuron<type_config>* all_neurons;

    value_type* working_memory_1;
    value_type* working_memory_2;

    __host__ __device__ void compute(const value_type* input, value_type* output);
};

struct SMALLEST_TYPES {
    using size_type = std::uint8_t;
    using index_type = std::uint8_t;
    using value_type = std::uint8_t;
};

template <typename type_config>
class network_owner {
public:
    using size_type = typename type_config::size_type;
    using neuron_t = neuron<type_config>;
    using layer_t = layer<type_config>;
    using network_t = network<type_config>;
    using value_type = typename type_config::value_type;

    network_owner(size_type input_size, size_type layer_count, const std::vector<size_type>& layer_sizes) {
        const auto neurons = get_total_neuron_count(layer_sizes);
        
        neurons_mem.resize(neurons);
        layers_mem.resize(layer_count);

        auto curr_neuron_offset = 0;
        size_type max_layer_size = 0;

        for(std::size_t layer = 0; layer < layer_count; ++layer) {
            layers_mem[layer].size = layer_sizes[layer];
            layers_mem[layer].input_size = (layer == 0) ? input_size : layer_sizes[layer - 1];
            layers_mem[layer].neurons = neurons_mem.data() + curr_neuron_offset;

            curr_neuron_offset += layer_sizes[layer];

            if (layer_sizes[layer] > max_layer_size) {
                max_layer_size = layer_sizes[layer];
            }
        }

        working_memory_1.resize(max_layer_size);
        working_memory_2.resize(max_layer_size);

        net.input_size = input_size;
        net.all_neurons = neurons_mem.data();
        net.layer_count = layer_count;
        net.layers = layers_mem.data();
        net.working_memory_1 = working_memory_1.data();
        net.working_memory_2 = working_memory_2.data();
    }

    network_t get_network_view() {
        return net;
    }

private:
    std::vector<neuron_t> neurons_mem;
    std::vector<layer_t> layers_mem;

    std::vector<value_type> working_memory_1;
    std::vector<value_type> working_memory_2;

    network_t net;

    size_type get_total_neuron_count(const std::vector<size_type>& layer_sizes) {
        size_type total = 0;
        for (const auto& size : layer_sizes) {
            total += size;
        }
        return total;
    }
};

}
#endif