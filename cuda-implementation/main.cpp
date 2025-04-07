#include "network.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

// Function to create a 3-bit adder network
template <typename type_config>
netw::network_owner<type_config> create_adder_network() {
    // Input size: 3 bits
    const typename type_config::size_type input_size = 3;
    
    // Define layer sizes - restructured for proper dependencies
    std::vector<typename type_config::size_type> layer_sizes = {
        3,  // First hidden layer: basic operations
        3,  // Second hidden layer: intermediate results
        2   // Output layer: final sum and carry
    };
    
    // Create the network
    netw::network_owner<type_config> network_owner(input_size, layer_sizes.size(), layer_sizes);
    auto network = network_owner.get_network_view();
    
    // Layer 1: Basic operations on inputs
    // - Neuron 0: a XOR b (part of sum logic)
    network.all_neurons[0].op = netw::op<type_config>::XOR;
    network.all_neurons[0].input1 = 0;  // Input a
    network.all_neurons[0].input2 = 1;  // Input b
    
    // - Neuron 1: a AND b (part of carry logic)
    network.all_neurons[1].op = netw::op<type_config>::AND;
    network.all_neurons[1].input1 = 0;  // Input a
    network.all_neurons[1].input2 = 1;  // Input b
    
    // - Neuron 2: Pass through input c
    network.all_neurons[2].op = netw::op<type_config>::NoOP;
    network.all_neurons[2].input1 = 2;  // Input c
    
    // Layer 2: Intermediate calculations
    // - Neuron 3: (a XOR b) XOR c (final sum bit)
    network.all_neurons[3].op = netw::op<type_config>::XOR;
    network.all_neurons[3].input1 = 0;  // a XOR b from layer 1
    network.all_neurons[3].input2 = 2;  // c from layer 1
    
    // - Neuron 4: (a XOR b) AND c (part of carry logic)
    network.all_neurons[4].op = netw::op<type_config>::AND;
    network.all_neurons[4].input1 = 0;  // a XOR b from layer 1
    network.all_neurons[4].input2 = 2;  // c from layer 1
    
    // - Neuron 5: Pass through a AND b
    network.all_neurons[5].op = netw::op<type_config>::NoOP;
    network.all_neurons[5].input1 = 1;  // a AND b from layer 1
    
    // Layer 3: Final output
    // - Neuron 6: Final sum bit (bit 0)
    network.all_neurons[6].op = netw::op<type_config>::NoOP;
    network.all_neurons[6].input1 = 0;  // (a XOR b) XOR c from layer 2
    
    // - Neuron 7: Final carry bit (bit 1): (a AND b) OR ((a XOR b) AND c)
    network.all_neurons[7].op = netw::op<type_config>::OR;
    network.all_neurons[7].input1 = 2;  // a AND b from layer 2
    network.all_neurons[7].input2 = 1;  // (a XOR b) AND c from layer 2
    
    return network_owner;
}

// Function to test the adder network
template <typename type_config>
void test_adder_network(netw::network<type_config>& network) {
    using value_type = typename type_config::value_type;
    
    std::cout << "Testing 3-bit Adder Network" << std::endl;
    std::cout << "==========================" << std::endl;
    std::cout << "  A + B + C = Result" << std::endl;
    std::cout << "--------------------------" << std::endl;
    
    // Allocate memory for input and output
    std::vector<value_type> input(network.input_size);
    std::vector<value_type> output(2);  // 2 output bits
    
    // Test all possible 3-bit inputs
    for (int a = 0; a <= 1; ++a) {
        for (int b = 0; b <= 1; ++b) {
            for (int c = 0; c <= 1; ++c) {
                // Set input values
                input[0] = a;
                input[1] = b;
                input[2] = c;
                
                // Compute network output
                network.compute(input.data(), output.data());
                
                // Calculate expected output
                int expected_sum = a + b + c;
                int expected_bit0 = expected_sum % 2;
                int expected_bit1 = expected_sum / 2;
                
                // Display result
                std::cout << "  " << a << " + " << b << " + " << c << " = " 
                          << output[1] << output[0]
                          << " (Expected: " << expected_bit1 << expected_bit0 << ")";
                
                // Verify result
                if (output[0] == expected_bit0 && output[1] == expected_bit1) {
                    std::cout << " ✓" << std::endl;
                } else {
                    std::cout << " ✗ ERROR!" << std::endl;
                }
            }
        }
    }
    std::cout << "==========================" << std::endl;
}

int main() {
    std::cout << "Binary Neural Network - 3-bit Adder Test" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    // Create a 3-bit adder network using small types
    auto network_owner = create_adder_network<netw::SMALLEST_TYPES>();
    auto network = network_owner.get_network_view();
    
    // Test the network
    test_adder_network(network);
    
    return 0;
}
