#!/usr/bin/env python3
import os
import re
import sys
import json
from collections import deque, defaultdict

def extract_accuracy(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
            # Find all occurrences of accuracy pattern
            accuracy_matches = re.findall(r'accuracy:\s*(\d+\.\d+)', content)
            if accuracy_matches:
                # Convert to float and return the highest accuracy
                accuracies = [float(match) for match in accuracy_matches]
                return max(accuracies)
    except Exception:
        pass
    return None

def extract_config_and_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
            # Find the configuration (parsed arguments)
            config_match = re.search(r'Parsed arguments:\s*(.+?)(?=\n)', content)
            config = config_match.group(1) if config_match else "No configuration found"
            
            # Find the best JSON network (the last one in the file)
            json_matches = re.findall(r'JSON:\s*(\{.+?\})\s*\n', content, re.DOTALL)
            if not json_matches:
                return config, None
            
            # Get the last JSON (usually the best one)
            json_str = json_matches[-1].strip()
            try:
                network_json = json.loads(json_str)
                return config, network_json
            except json.JSONDecodeError:
                return config, None
    except Exception as e:
        print(f"Error extracting JSON from {file_path}: {e}")
    return None, None

def find_useful_neurons(network):
    if not network:
        return 0, 0
    
    # Initialize data structures
    layers = network['layers']
    connected_neurons = set()
    total_neurons = 0
    
    # Start with output neurons and work backward
    output_layer_idx = len(layers) - 1
    output_neurons = [(output_layer_idx, i) for i in range(len(layers[output_layer_idx]['neurons']))]
    
    # Initialize queue for BFS
    queue = deque(output_neurons)
    
    # Mark all output neurons as connected
    for layer_idx, neuron_idx in output_neurons:
        connected_neurons.add((layer_idx, neuron_idx))
    
    # BFS to find all neurons connected to outputs
    while queue:
        layer_idx, neuron_idx = queue.popleft()
        neuron = layers[layer_idx]['neurons'][neuron_idx]
        
        # Skip input layer (for traversal purposes)
        if layer_idx == 0:
            continue
        
        # Check all input connections to this neuron
        for arg_idx in neuron['arguments_indexes']:
            # If the argument index is less than input size, it refers to a neuron in the previous layer
            if arg_idx < layers[layer_idx]['input_size']:
                prev_layer_idx = layer_idx - 1
                
                # If already processed, skip
                src_neuron_id = (prev_layer_idx, arg_idx)
                if src_neuron_id in connected_neurons:
                    continue
                
                # Mark source neuron as connected
                connected_neurons.add(src_neuron_id)
                
                # Add to queue to process its inputs
                queue.append((prev_layer_idx, arg_idx))
    
    # Count total neurons and calculate operational cost
    total_neurons = 0
    operational_cost = 0
    
    # Go through each layer and count operations for connected neurons
    for layer_idx, layer in enumerate(layers):
        total_neurons += len(layer['neurons'])
        
        # For each neuron in this layer
        for neuron_idx, neuron in enumerate(layer['neurons']):
            # Check if this neuron is connected
            if (layer_idx, neuron_idx) in connected_neurons:
                # If it's not a NoOp and not in the input layer, count it for operational cost
                if neuron['operation'] != "NoOp":
                    operational_cost += 1
    
    return len(connected_neurons), operational_cost

def find_best_accuracy_files(directory):
    results = []
    
    # Walk through all files in the directory
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            accuracy = extract_accuracy(file_path)
            if accuracy is not None:
                results.append((file_path, accuracy))
    
    # Sort by accuracy (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Return top 20 or all if less than 20
    # return results[:20]
    return results

def main():
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print("Usage: python find-best-accuracy.py <directory> <optional: display info")
        sys.exit(1)
    
    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory")
        sys.exit(1)
    
    display_info = len(sys.argv) == 3 and sys.argv[2] == "info"

    top_files = find_best_accuracy_files(directory)
    
    if not top_files:
        print("No files with accuracy information found.")
        return
    
    print("Top files with highest accuracy:")
    for file_path, accuracy in top_files:
        print(f"{file_path} - accuracy: {accuracy}", end='')
        
        # Extract configuration and network JSON
        config, network_json = extract_config_and_json(file_path)

        if display_info:
            print(f"\n  Configuration: {config}")
        
        if network_json:
            # Calculate and print useful neurons and operational cost
            useful_count, operational_cost = find_useful_neurons(network_json)
            total_neurons = sum(len(layer['neurons']) for layer in network_json['layers'])
            
            if display_info:
                print(f"  Total neurons: {total_neurons}")
                print(f"  Useful neurons: {useful_count}")
                print(f"  Operational cost: {operational_cost}\n")
            else:
                print(f', cost: {operational_cost}')
            
        else:
            print("Could not extract network JSON from the file.")

if __name__ == "__main__":
    main()
