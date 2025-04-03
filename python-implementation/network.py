from itertools import product
import json
import random
random.seed(4200)

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

class Metrics:
    """Collection of standard evaluation metrics for neural networks."""
    
    @staticmethod
    def accuracy(predicted_outputs, expected_outputs):
        """Calculate accuracy (correct predictions / total predictions)"""
        correct = 0
        total = len(predicted_outputs)
        
        for i in range(total):
            if predicted_outputs[i] == expected_outputs[i]:
                correct += 1
                
        return correct / total if total > 0 else 0
    
    @staticmethod
    def _check_binary_output(predicted_outputs, expected_outputs):
        """Check if outputs are binary for binary classification metrics"""
        # Flatten outputs if they're lists of lists with single elements
        if predicted_outputs and isinstance(predicted_outputs[0], list) and len(predicted_outputs[0]) == 1:
            flat_predicted = [p[0] for p in predicted_outputs]
        else:
            flat_predicted = predicted_outputs
            
        if expected_outputs and isinstance(expected_outputs[0], list) and len(expected_outputs[0]) == 1:
            flat_expected = [e[0] for e in expected_outputs]
        else:
            flat_expected = expected_outputs
            
        return flat_predicted, flat_expected
    
    @staticmethod
    def precision(predicted_outputs, expected_outputs):
        """Calculate precision (true positives / (true positives + false positives))"""
        predicted, expected = Metrics._check_binary_output(predicted_outputs, expected_outputs)
        
        true_positives = 0
        false_positives = 0
        
        for i in range(len(predicted)):
            if predicted[i] == 1:
                if expected[i] == 1:
                    true_positives += 1
                else:
                    false_positives += 1
                    
        return true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    
    @staticmethod
    def recall(predicted_outputs, expected_outputs):
        """Calculate recall (true positives / (true positives + false negatives))"""
        predicted, expected = Metrics._check_binary_output(predicted_outputs, expected_outputs)
        
        true_positives = 0
        false_negatives = 0
        
        for i in range(len(predicted)):
            if expected[i] == 1:
                if predicted[i] == 1:
                    true_positives += 1
                else:
                    false_negatives += 1
                    
        return true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    @staticmethod
    def f1_score(predicted_outputs, expected_outputs):
        """Calculate F1 score (2 * (precision * recall) / (precision + recall))"""
        precision = Metrics.precision(predicted_outputs, expected_outputs)
        recall = Metrics.recall(predicted_outputs, expected_outputs)
        
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

class NetworkEvaluator:
    def __init__(self):
        self.network = None
        self.input_values = []
        self.expected_outputs = []
        self.evaluation_metric = Metrics.accuracy
        self.statistics_metrics = {
            'accuracy': Metrics.accuracy,
            'precision': Metrics.precision,
            'recall': Metrics.recall,
            'f1_score': Metrics.f1_score
        }

    def set_network(self, network: Network):
        self.network = network
        return self

    def set_inputs(self, input_values: list[list[int]], expected_outputs: list[list[int]]):
        self.input_values = input_values
        self.expected_outputs = expected_outputs
        return self

    def set_inputs_based_on_function(self, function: callable, input_size: int):
        self.input_values = list([list(x) for x in product([0, 1], repeat=input_size)])
        self.expected_outputs = list([function(inputs) for inputs in self.input_values])
        return self
    
    def set_evaluation_metric(self, metric_function: callable):
        self.evaluation_metric = metric_function
        return self
    
    def configure_statistics(self, metrics_dict: dict[str, callable]):
        self.statistics_metrics = metrics_dict
        return self
    
    def evaluate(self) -> float:
        predicted_outputs = []
        
        for inputs in self.input_values:
            output = self.network.evaluate(inputs)
            predicted_outputs.append(output)
            
        return self.evaluation_metric(predicted_outputs, self.expected_outputs)
    
    def get_statistics(self) -> str:
        if not self.statistics_metrics:
            return "No statistics metrics configured."
        
        predicted_outputs = []
        for inputs in self.input_values:
            output = self.network.evaluate(inputs)
            predicted_outputs.append(output)
        
        results = []
        for name, metric_function in self.statistics_metrics.items():
            try:
                value = metric_function(predicted_outputs, self.expected_outputs)
                results.append(f"{name}: {value:.4f}")
            except Exception as e:
                results.append(f"{name}: Error - {str(e)}")
                
        return "\n".join(results)
    
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

class StochasticExplorationConfig:
    def __init__(self):
        self.samples_per_distance: list[tuple[int, int]] = []

    def set_samples_per_distance(self, distance: int, samples: int):
        self.samples_per_distance.append((distance, samples))
        return self
    
    def get_samples_per_distance(self) -> list[tuple[int, int]]:
        return sorted(self.samples_per_distance, key=lambda x: x[0])
    

class GradientDescent:
    def configure(self,
                  network: Network,
                  evaluator: NetworkEvaluator,
                  mix_up_coefficient: float = 0.1,
                  stochastic_exploration_config: StochasticExplorationConfig = None):
        
        self.network: Network = network
        self.evaluator: NetworkEvaluator = evaluator.set_network(network)
        self.mix_up_coefficient = mix_up_coefficient
        self.stochastic_exploration_config: StochasticExplorationConfig = stochastic_exploration_config

        return self

    def run(self, max_steps: int = 1000000):
        best_score = 0
        steps = 0
        did_step = True
        
        while best_score < 1 and steps < max_steps:
            stats = self.evaluator.get_statistics().replace('\n', ', ')
            print(f"Step {steps} -- stats - {stats}")
            
            if not did_step:
                # print(" -- Refreshing network, network score was:", self.evaluator.evaluate())
                # self._refresh_network()
                print(" -- Mixing up the network, network stats -", self.evaluator.get_statistics().replace('\n', ', '))
                self._mix_up_the_network()

            did_step = self._do_step()
            score = self.evaluator.evaluate()

            if score > best_score:
                best_score = score

                stats = self.evaluator.get_statistics().replace('\n', ', ')
                print (f"New best score - {best_score} (stats - {stats}), network:")
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
        one_step_successful = self._do_one_change_step()
        if one_step_successful:
            return True
        
        print(" -- One step was not successful, trying stochastic exploration")

        stochastic_step_successful = self._do_stochastic_exploration_steps()

        if stochastic_step_successful:
            print(" -- Stochastic step was successful")

        return stochastic_step_successful

    def _do_one_change_step(self):
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
    
    def _do_stochastic_exploration_steps(self):
        if self.stochastic_exploration_config is None:
            return False
        
        random_editor = NetworkRandomEditor().set_network(self.network)
        best_score = self.evaluator.evaluate()
        best_changed_score = 0
        best_changes = None

        for distance, samples in self.stochastic_exploration_config.get_samples_per_distance():
            for _ in range(samples):
                random_editor.do_change_to_the_network(distance)

                score = self.evaluator.evaluate()
                if score > best_score:
                    best_score = score
                    best_changes = random_editor.get_changes()
                    print (" -- found better model:", self.evaluator.get_statistics().replace('\n', ', '))

                if score > best_changed_score:
                    best_changed_score = score

                random_editor.undo_changes()

        if best_changes is not None:
            best_changes.apply(self.network)
            return True
        else:
            print(" -- No better model found, best score was:", best_changed_score)
        
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
    network=Network(7, [20, 15, 9, 5, 3, 2, 1]),
    # network=Network(7, [15, 9, 5, 3, 2, 1]),
    # network=Network(7, [3, 2, 1]),
    evaluator=NetworkEvaluator().set_inputs_based_on_function(three_two_bits_and_center, 7),
    # mix_up_coefficient=.03,
    mix_up_coefficient=.05,
    stochastic_exploration_config=StochasticExplorationConfig()
        .set_samples_per_distance(2, 10_000)
        .set_samples_per_distance(3,  8_000)
        .set_samples_per_distance(4,  5_000)
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