import json
import random
from network import Network
from evaluation import NetworkEvaluator
from sgd import GradientDescent, StochasticExplorationConfig
from arg_parsing import parse_arguments

def three_two_bits_and_center(input_values: list[int]) -> int:
    """Conway's Game of Life rule: a cell is alive if it has exactly 3 neighbors,
    or if it was already alive and has 2 neighbors"""
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

def main():
    args = parse_arguments()
    print("Parsed arguments:", args)

    if args.seed is not None:    
        random.seed(args.seed)
    
    function_map = {
        'game_of_life': three_two_bits_and_center,
    }
    
    chosen_function = function_map[args.function]
    input_size = args.input_size
    
    stochastic_config = None
    if args.stochastic:
        stochastic_config = StochasticExplorationConfig()
        for distance, samples in args.distance_samples:
            stochastic_config.set_samples_per_distance(distance, samples)
    
    GradientDescent().configure(
        network=Network(input_size, args.layers),
        evaluator=NetworkEvaluator().set_inputs_based_on_function(chosen_function, input_size),
        mix_up_coefficient=args.mix_up,
        stochastic_exploration_config=stochastic_config
    ).run(
        max_steps=args.max_steps
    )

if __name__ == "__main__":
    main()
