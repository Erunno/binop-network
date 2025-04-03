import argparse
from typing import List, Tuple

def parse_arguments():
    parser = argparse.ArgumentParser(description='Binary Operations Network Training')
    
    # Network architecture
    parser.add_argument('--layers', nargs='+', type=int, 
                        default=[20, 15, 9, 5, 3, 2, 1],
                        help='Sizes of network layers (space-separated integers)')
    
    # Input size
    parser.add_argument('--input-size', type=int, default=7,
                        help='Size of input vector')
    
    # Training parameters
    parser.add_argument('--max-steps', type=int, default=1_000_000,
                        help='Maximum number of training steps')
    parser.add_argument('--mix-up', type=float, default=0.1,
                        help='Mix-up coefficient for network perturbation')
    
    # Random seed
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    # Stochastic exploration parameters
    parser.add_argument('--stochastic', action='store_true', default=True,
                        help='Enable stochastic exploration')
    parser.add_argument('--no-stochastic', action='store_false', dest='stochastic',
                        help='Disable stochastic exploration')
    
    # Distances and samples for stochastic exploration
    parser.add_argument('--distance-samples', nargs='+', type=int,
                        default=None,
                        help='Distance and sample counts for stochastic exploration. Format: distance samples [distance samples ...]')

    # Choose function
    parser.add_argument('--function', choices=['game_of_life'],
                        default='game_of_life',
                        help='Function to learn')
    
    args = parser.parse_args()
    
    # Process distance-samples
    if args.distance_samples is None:
        args.distance_samples = [(2, 10000), (3, 8000), (4, 5000)]
    else:
        # Convert flat list to list of tuples (distance, samples)
        pairs = []
        for i in range(0, len(args.distance_samples), 2):
            if i + 1 < len(args.distance_samples):
                pairs.append((args.distance_samples[i], args.distance_samples[i+1]))
        args.distance_samples = pairs
    
    return args
