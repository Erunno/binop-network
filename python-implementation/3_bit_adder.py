import json
import random
from network import Network
from evaluation import NetworkEvaluator
from sgd import GradientDescent, StochasticExplorationConfig
from arg_parsing import parse_arguments

def add_3(input_values: list[int]) -> int:
    sum = input_values[0] + input_values[1] + input_values[2]
    return [ sum % 2, sum // 2 ]


def main():
    input_size = 3
    layers = [3, 2, 2]

    GradientDescent().configure(
        network=Network(input_size, layers),
        evaluator=NetworkEvaluator() \
            .set_inputs_based_on_function(add_3, input_size) \
            .set_evaluation_metric('mcc'),
        mix_up_coefficient=0.3,
        stochastic_exploration_config=StochasticExplorationConfig() \
            .set_samples_per_distance(2, 1000) \
            .set_samples_per_distance(3, 1000),
    ).run(
        max_steps=10000
    )

for a in [0, 1]:
    for b in [0, 1]:
        for c in [0, 1]:
            _0 = a ^ b ^ c;
            _1 = (a & b) | ((a ^ b) & c);

            ref_0, ref_1 = add_3([a, b, c])

            print(f"{a} + {b} + {c} = {_0} {_1} -- {'✅' if ref_0 == _0 and ref_1 == _1 else '❌'}")
exit(0)
if __name__ == "__main__":
    main()