import json
import random
from network import Network
from evaluation import NetworkEvaluator
from sgd import GradientDescent, StochasticExplorationConfig

# Initialize random seed for reproducibility
random.seed(420)

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
    mix_up_coefficient=.1,
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