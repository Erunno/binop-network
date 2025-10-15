from itertools import product
import json
import random
from operations import Function, And, Or, Not, Xor, NoOp

class Layer:
    def __init__(self, prev_layer_size: int, size: int, use_random: bool = True):
        self.functions = Layer.get_all_functions(prev_layer_size)
        
        self.size = size
        self.input_size = prev_layer_size

        if use_random:
            self.neurons = Layer.get_random_array_of_ints(size, len(self.functions) - 1)
        else:
            self.neurons = [0] * size

    def eval(self, input_values: list[int]) -> list[int]:
        if len(input_values) != self.input_size:
            raise ValueError(f"Input values size does not match the number of functions ({len(self.neurons)}), got {len(input_values)})")
        
        output = []
        for neuron_idx in self.neurons:
            function = self.functions[neuron_idx]
            output.append(function.eval(input_values))
        return output
    
    def __str__(self):
        function_strs = [str(self.functions[neuron_idx]) for neuron_idx in self.neurons]
        return f"Layer(size={self.size}, neurons=[{', '.join(function_strs)})]"
    
    def refresh(self):
        for i in range(self.size):
            self.neurons[i] = random.randint(0, len(self.functions) - 1)

    def mix_up(self, change_coefficient: float):
        for i in range(self.size):
            if random.random() < change_coefficient:
                self.neurons[i] = random.randint(0, len(self.functions) - 1)

    def json(self):
        return {
            'input_size': self.input_size,
            'neurons': [self.functions[neuron_idx].json() for neuron_idx in self.neurons],
        }
    
    @staticmethod
    def get_random_array_of_ints(size: int, max_val) -> list[int]:
        return [random.randint(0, max_val) for _ in range(size)]
    
    @staticmethod
    def get_all_functions(input_size: int) -> list[Function]:
        ands = [And(i, j) for i in range(input_size) for j in range(i + 1, input_size) if i != j]
        ors = [Or(i, j) for i in range(input_size) for j in range(i + 1, input_size) if i != j]
        xors = [Xor(i, j) for i in range(input_size) for j in range(i + 1, input_size) if i != j]
        nots = [Not(i) for i in range(input_size)]
        no_ops = [NoOp(i) for i in range(input_size)]

        return ands + ors + nots + xors + no_ops


class Network:
    def __init__(self, input_size, layer_sizes: list[int], use_random: bool = True):
        self.layers: list[Layer] = []
        self.input_size = input_size
        prev_layer_size = input_size

        for size in layer_sizes:
            layer = Layer(prev_layer_size, size, use_random)
            self.layers.append(layer)
            prev_layer_size = size
    
    def evaluate(self, input_values: list[int]) -> list[int]:
        if len(input_values) != self.input_size:
            raise ValueError(f"Input values size does not match the number of functions in the first layer ({self.input_size}), got {len(input_values)})")
        
        for layer in self.layers:
            input_values = layer.eval(input_values)

        return input_values

    def evaluate_get_values_of_all(self, input_values: list[int]) -> list[list[int]]:
        if len(input_values) != self.input_size:
            raise ValueError(f"Input values size does not match the number of functions in the first layer ({self.input_size}), got {len(input_values)})")
        
        values = [[*input_values]]

        for layer in self.layers:
            input_values = layer.eval(input_values)
            values.append([*input_values])

        return values

    def refresh(self):
        for layer in self.layers:
            layer.refresh()

    def mix_up(self, change_coefficient):
        for layer in self.layers:
            layer.mix_up(change_coefficient)

    def json(self):
        return {
            'input_size': self.input_size,
            'layers': [layer.json() for layer in self.layers],
        }

    def __str__(self):
        ret_val = 'Network(\n'
        ret_val += f"  Input size: {self.input_size}\n"

        for i, layer in enumerate(self.layers):
            ret_val += f"  {layer}\n"

        return ret_val + ')'

    @staticmethod
    def from_json(json_string: str) -> 'Network':
        data = json.loads(json_string)
        input_size = data['input_size']
        layers = []

        def find_index(functions: list[Function], function: Function) -> int:
            for i, f in enumerate(functions):
                if f.is_same(function):
                    return i
            raise ValueError("Function not found in the list")

        for layer_data in data['layers']:
            layer = Layer(layer_data['input_size'], len(layer_data['neurons']), use_random=False)
            neuron_functions = [Function.from_json(json.dumps(neuron)) for neuron in layer_data['neurons']]
            layer.neurons = [find_index(layer.functions, neuron_func) for neuron_func in neuron_functions]
            layers.append(layer)

        network = Network(input_size, [layer.size for layer in layers], False)
        network.layers = layers

        return network

    def get_pruned(self):
        """
        Prunes the network by removing neurons that aren't connected to the output.
        Returns a new Network with only the neurons that influence the output.
        """
        # Create a deep copy of the network structure
        original_network = json.loads(json.dumps(self.json()))
        
        # Track connected neurons in each layer
        connected_neurons = {}
        for i in range(len(original_network['layers'])):
            connected_neurons[i] = set()
        
        # Start with all output neurons (last layer) marked as connected
        output_layer_index = len(original_network['layers']) - 1
        for i in range(len(original_network['layers'][output_layer_index]['neurons'])):
            connected_neurons[output_layer_index].add(i)
        
        # Trace backward from outputs to inputs to find connected neurons
        for layer_index in range(output_layer_index, 0, -1):
            layer = original_network['layers'][layer_index]
            prev_layer_index = layer_index - 1
            
            # For each connected neuron in this layer
            for neuron_index in connected_neurons[layer_index]:
                neuron = layer['neurons'][neuron_index]
                
                # Mark all input sources to this neuron as connected
                for arg_index in neuron.get('arguments_indexes', []):
                    connected_neurons[prev_layer_index].add(arg_index)
        
        # Create the pruned network structure
        pruned_network = {
            'input_size': original_network['input_size'],
            'layers': []
        }
        
        # Keep track of index mappings from original to pruned network
        index_mappings = {}
        
        # Process each layer to build the pruned network
        for layer_index in range(len(original_network['layers'])):
            original_layer = original_network['layers'][layer_index]
            
            # Create mapping from original indices to new indices
            index_mappings[layer_index] = {}
            new_neurons = []
            new_index = 0
            
            # For each neuron in this layer
            for i in range(len(original_layer['neurons'])):
                # Only include if it's connected to outputs
                if i in connected_neurons[layer_index]:
                    # Record the mapping of original index to new index
                    index_mappings[layer_index][i] = new_index
                    new_index += 1
                    
                    # Clone the neuron
                    original_neuron = original_layer['neurons'][i]
                    new_neuron = json.loads(json.dumps(original_neuron))  # Deep copy
                    
                    # Remap argument indexes for non-first layers
                    if layer_index > 0 and 'arguments_indexes' in new_neuron:
                        new_neuron['arguments_indexes'] = [
                            index_mappings[layer_index - 1].get(arg_idx, 0)  # Fallback to 0 if not found
                            for arg_idx in new_neuron['arguments_indexes']
                        ]
                    
                    new_neurons.append(new_neuron)
            
            # Create the new layer
            new_layer = {
                'input_size': original_network['input_size'] if layer_index == 0 
                               else len(pruned_network['layers'][layer_index - 1]['neurons']),
                'neurons': new_neurons
            }
            
            pruned_network['layers'].append(new_layer)
        
        # Convert the pruned network JSON back to a Network object
        return Network.from_json(json.dumps(pruned_network))