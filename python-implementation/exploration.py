import random
from network import Network, Layer

class NetworkExplorer:
    def set_network(self, network: Network):
        self.network: Network = network

        self.layer_idx = 0
        self.neuron_idx = 0
        self.explored_function_count = 0

        self.explored_all = False

        return self

    def move_next(self):
        layer = self.network.layers[self.layer_idx]

        function_count_in_layer = len(layer.functions)

        layer.neurons[self.neuron_idx] = (layer.neurons[self.neuron_idx] + 1) % function_count_in_layer
        self.explored_function_count += 1

        if self.explored_function_count < function_count_in_layer:
            return
        
        self.explored_function_count = 0
        self.neuron_idx += 1

        if self.neuron_idx < layer.size:
            self.move_next()
            return
        
        self.neuron_idx = 0
        self.layer_idx += 1

        if self.layer_idx < len(self.network.layers):
            self.move_next()
            return
        
        self.layer_idx = 0

        self.explored_all = True

    class State:
        def __init__(self, layer_idx, neuron_idx, explored_function_count):
            self.layer_idx = layer_idx
            self.neuron_idx = neuron_idx
            self.explored_function_count = explored_function_count

    def export_state(self):
        return NetworkExplorer.State(self.layer_idx, self.neuron_idx, self.explored_function_count)
    
    def adjust_network(self, state: 'NetworkExplorer.State'):
        layer = self.network.layers[state.layer_idx]
        function_offset = state.explored_function_count

        layer.neurons[state.neuron_idx] = (layer.neurons[state.neuron_idx] + function_offset) % len(layer.functions)


class NetworkRandomEditor:
    class NeuronStates:
        def __init__(self, ):
            self.states = []

        def save(self, layer_idx: int, neuron_idx: int, function_idx: int):
            self.states.append((layer_idx, neuron_idx, function_idx))
            
        def apply(self, network: Network):
            for layer_idx, neuron_idx, function_idx in reversed(self.states):
                layer: Layer = network.layers[layer_idx]
                layer.neurons[neuron_idx] = function_idx

        def clear(self):
            self.states = []

        def clone(self):
            clone = NetworkRandomEditor.NeuronStates()
            clone.states = list([state for state in self.states])

            return clone

    def set_network(self, network: Network):
        self.network = network
        self.last_state = NetworkRandomEditor.NeuronStates()
        self.last_performed_changes = NetworkRandomEditor.NeuronStates()

        return self

    def do_change_to_the_network(self, changed_neurons_count: int):
        for _ in range(changed_neurons_count):
            layer_idx = random.randint(0, len(self.network.layers) - 1)
            layer = self.network.layers[layer_idx]

            neuron_idx = random.randint(0, layer.size - 1)

            changed_function_idx = random.randint(0, len(layer.functions) - 1)
            old_function_idx = layer.neurons[neuron_idx]

            self.last_state.save(layer_idx, neuron_idx, old_function_idx)
            self.last_performed_changes.save(layer_idx, neuron_idx, changed_function_idx)

            layer.neurons[neuron_idx] = changed_function_idx

        return self
    
    def undo_changes(self):
        self.last_state.apply(self.network)

        self.last_state.clear()
        self.last_performed_changes.clear()

        return self
    
    def get_changes(self) -> 'NetworkRandomEditor.NeuronStates':
        return self.last_performed_changes.clone()
