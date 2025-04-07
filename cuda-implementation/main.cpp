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

// Function to test the network with temporary neuron changes
template <typename type_config>
void test_with_changed_neurons(netw::network<type_config>& network) {
    using value_type = typename type_config::value_type;
    
    std::cout << "\nTesting with Temporarily Changed Neurons" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    // Create a change list
    std::vector<netw::neuron_change<type_config>> changes_vec(2); // 2 changes
    
    // Modify the first hidden layer's XOR to OR
    changes_vec[0].layer_index = 0;
    changes_vec[0].neuron_index = 0;
    changes_vec[0].op = netw::op<type_config>::OR;
    changes_vec[0].input1 = 0;
    changes_vec[0].input2 = 1;
    
    // Modify the second hidden layer's XOR to AND
    changes_vec[1].layer_index = 1;
    changes_vec[1].neuron_index = 0;
    changes_vec[1].op = netw::op<type_config>::AND;
    changes_vec[1].input1 = 0;
    changes_vec[1].input2 = 2;
    
    // Create a change list
    netw::neuron_change_list<type_config> changes;
    changes.count = 2;
    changes.changes = changes_vec.data();
    
    // Test a specific input with the modified network
    std::vector<value_type> input = {1, 0, 1};  // Testing [1,0,1]
    std::vector<value_type> output(2);
    
    // Compute with original network
    network.compute(input.data(), output.data());
    std::cout << "Original network result for [1,0,1]: "
              << static_cast<int>(output[1]) << static_cast<int>(output[0]) << std::endl;
    
    // Compute with changed neurons (temporary changes)
    network.compute_with_changed_neurons(changes, input.data(), output.data());
    std::cout << "Modified network result for [1,0,1]: "
              << static_cast<int>(output[1]) << static_cast<int>(output[0]) << std::endl;
    
    // Verify the network hasn't been modified by computing again with original network
    network.compute(input.data(), output.data());
    std::cout << "Original network result after temp change: "
              << static_cast<int>(output[1]) << static_cast<int>(output[0]) << std::endl;
              
    std::cout << "=====================================" << std::endl;
}

// Function to test the network after applying permanent changes
template <typename type_config>
void test_with_applied_changes(netw::network<type_config>& network) {
    using value_type = typename type_config::value_type;
    
    std::cout << "\nTesting with Permanently Applied Changes" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    // Create a change list
    std::vector<netw::neuron_change<type_config>> changes_vec(1); // 1 change
    
    // Modify the first hidden layer's XOR to AND
    changes_vec[0].layer_index = 0;
    changes_vec[0].neuron_index = 0;
    changes_vec[0].op = netw::op<type_config>::AND;
    changes_vec[0].input1 = 0;
    changes_vec[0].input2 = 1;
    
    // Create a change list
    netw::neuron_change_list<type_config> changes;
    changes.count = 1;
    changes.changes = changes_vec.data();
    
    // Test input [1,1,0]
    std::vector<value_type> input = {1, 1, 0};
    std::vector<value_type> output(2);
    
    // Compute with original network
    network.compute(input.data(), output.data());
    std::cout << "Result before applying changes [1,1,0]: "
              << static_cast<int>(output[1]) << static_cast<int>(output[0]) << std::endl;
    
    // Apply changes permanently
    changes.apply_changes(&network);
    std::cout << "Changes permanently applied to network" << std::endl;
    
    // Compute with modified network (changes are now permanent)
    network.compute(input.data(), output.data());
    std::cout << "Result after applying changes [1,1,0]: "
              << static_cast<int>(output[1]) << static_cast<int>(output[0]) << std::endl;
              
    // Test with more inputs to show the changes are indeed permanent
    input = {1, 0, 1};
    network.compute(input.data(), output.data());
    std::cout << "New result for [1,0,1] after permanent change: "
              << static_cast<int>(output[1]) << static_cast<int>(output[0]) << std::endl;
              
    std::cout << "=====================================" << std::endl;
}

int main() {
    std::cout << "Binary Neural Network - 3-bit Adder Test" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    // Create a 3-bit adder network using small types
    auto network_owner = create_adder_network<netw::SMALLEST_TYPES>();
    auto network = network_owner.get_network_view();
    
    // First, test the original network
    test_adder_network(network);
    
    // Test with temporary changes
    test_with_changed_neurons(network);
    
    // Test with permanent changes
    test_with_applied_changes(network);
    
    return 0;
}
