from itertools import product
import json
import random
random.seed(420)

class Function:
    def eval(self, input_values: list[int]) -> int:
        raise NotImplementedError("This method should be overridden by subclasses")

class BinOp(Function):
    def __init__(self, fst_neuron_idx, sec_neuron_idx):
        self.fst_neuron_idx = fst_neuron_idx
        self.sec_neuron_idx = sec_neuron_idx

    def eval(self, input_values: list[int]) -> int:
        fst_value = input_values[self.fst_neuron_idx]
        sec_value = input_values[self.sec_neuron_idx]

        return self.op(fst_value, sec_value)
    
    def __str__(self):
        return f"{self.__class__.__name__}({self.fst_neuron_idx}, {self.sec_neuron_idx})"
    
    def json(self):
        return {
            'operation': self.__class__.__name__,
            'arguments_indexes': [
                self.fst_neuron_idx,
                self.sec_neuron_idx
            ]
        }
    
class UnOp(Function):
    def __init__(self, neuron_idx):
        self.fst_neuron_idx = neuron_idx

    def eval(self, input_values: list[int]) -> int:
        return self.op(input_values[self.fst_neuron_idx])

    def __str__(self):
        return f"{self.__class__.__name__}({self.fst_neuron_idx})"

    def json(self):
        return {
            'operation': self.__class__.__name__,
            'arguments_indexes': [
                self.fst_neuron_idx
            ]
        }
    

class And(BinOp):
    def op(self, fst_value, sec_value):
        return fst_value & sec_value

class Or(BinOp):
    def op(self, fst_value, sec_value):
        return fst_value | sec_value
    
class Not(UnOp):
    def op(self, value):
        return 1 if value == 0 else 0
    
class Xor(BinOp):
    def op(self, fst_value, sec_value):
        return 1 if fst_value != sec_value else 0

class NoOp(UnOp):
    def op(self, value):
        return value

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

class NetworkEvaluator:
    def set_network(self, network: Network):
        self.network = network

        self.layer_idx = 0
        self.neuron_idx = 0

        return self

    def set_inputs(self, input_values: list[list[int]], expected_outputs: list[list[int]]):
        self.input_values = input_values
        self.expected_outputs = expected_outputs

        return self

    def set_inputs_based_on_function(self, function: callable, input_size: int):
        self.input_values = list([list(x) for x in product([0, 1], repeat=input_size)])
        self.expected_outputs = list([function(inputs) for inputs in self.input_values])
        return self

    def evaluate(self) -> list[int]:
        correct = 0

        for i in range(len(self.input_values)):
            inputs = self.input_values[i]
            expected = self.expected_outputs[i]

            output = self.network.evaluate(inputs)

            if output == expected:
                correct += 1

        return correct / len(self.input_values)
    

    def test_data_json(self):
        zipped = zip(self.input_values, self.expected_outputs)

        return {
            'data': [
                {
                    'input': input_values,
                    'expected_output': expected_output
                } for input_values, expected_output in zipped
            ]
        }
    
class GradientDescent:
    
    def configure(self,
                  network: Network,
                  evaluator: NetworkEvaluator,
                  mix_up_coefficient: float = 0.1):
        self.network: Network = network
        self.evaluator: NetworkEvaluator = evaluator.set_network(network)
        self.mix_up_coefficient = mix_up_coefficient

        return self

    def run(self, max_steps: int = 1000000):
        best_score = 0
        steps = 0
        did_step = True
        
        while best_score < 1 and steps < max_steps:
            print(f"Step {steps}:")
            
            if not did_step:
                # print(" -- Refreshing network, network score was:", self.evaluator.evaluate())
                # self._refresh_network()
                print(" -- Mixing up the network, network score was:", self.evaluator.evaluate())
                self._mix_up_the_network()

            did_step = self._do_step()
            score = self.evaluator.evaluate()

            if score > best_score:
                best_score = score
                
                print (f"New best score: {best_score}, network:")
                print(self.network)
                print("JSON:")
                print(json.dumps(self.network.json()))
                print()

            steps += 1

        if steps >= max_steps:
            print("Max steps reached")

        print(f"Final score: {best_score}")
        print("Final network:")
        print(self.network)

    def _do_step(self):
        explorer = NetworkExplorer().set_network(self.network)
        best_state = None
        best_score = self.evaluator.evaluate()

        while True:
            explorer.move_next()
            
            if explorer.explored_all:
                break

            score = self.evaluator.evaluate()

            if score > best_score:
                best_score = score
                best_state = explorer.export_state()

        if best_state is not None:
            explorer.adjust_network(best_state)
            return True

        return False
    
    def _refresh_network(self):
        self.network.refresh()

    def _mix_up_the_network(self):
        self.network.mix_up(change_coefficient=self.mix_up_coefficient)

def add_three_bits(input_values: list[int]) -> int:
    if len(input_values) != 3:
        raise ValueError("Input values must be a list of 3 integers")
    
    sum = input_values[0] + input_values[1] + input_values[2]

    return [sum // 2, sum % 2]

# GradientDescent().configure(
#     network=Network(3, [3, 2, 2]),
#     evaluator=NetworkEvaluator().set_inputs_based_on_function(add_three_bits, 3),
#     mix_up_coefficient=.4
# ).run(
#     max_steps=1000
# )


def three_two_bits_and_center(input_values: list[int]) -> int:
    if len(input_values) != 7:
        raise ValueError("Input values must be a list of 7 integers")
    
    A_0 = input_values[0]
    A_1 = input_values[1]
    B_0 = input_values[2]
    B_1 = input_values[3]
    C_0 = input_values[4]
    C_1 = input_values[5]

    center = input_values[6]

    sum = A_0 + 2*A_1 + B_0 + 2*B_1 + C_0 + 2*C_1

    should_be_alive = sum == 3 or (center == 1 and sum == 2)

    return [1 if should_be_alive else 0]

# test_data = NetworkEvaluator().set_inputs_based_on_function(three_two_bits_and_center, 7).test_data_json()
# print(json.dumps(test_data))
# exit()

eval = NetworkEvaluator().set_inputs_based_on_function(three_two_bits_and_center, 7)

GradientDescent().configure(
    network=Network(7, [15, 9, 5, 3, 2, 1]),
    evaluator=NetworkEvaluator().set_inputs_based_on_function(three_two_bits_and_center, 7),
    mix_up_coefficient=.03
).run(
    max_steps=1000
)


def sum_and_alive(input_values: list[int]) -> int:
    if len(input_values) != 4:
        raise ValueError("Input values must be a list of 4 integers")
    
    sum = input_values[0] + input_values[1] * 2 + input_values[2] * 4
    center = input_values[3]

    should_be_alive = sum == 3 or (center == 1 and sum == 2)
    return [1 if should_be_alive else 0]

# GradientDescent().configure(
#     network=Network(4, [3, 2, 1]),
#     evaluator=NetworkEvaluator().set_inputs_based_on_function(sum_and_alive, 4)
# ).run(
#     max_steps=1000
# )