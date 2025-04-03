# BinOp Network 🧠⚙️

A neural network implementation that uses binary operations as its building blocks. This project includes both a Python implementation for training networks and an interactive visualization tool.

## Features ✨

- 🧮 Neural network built with binary operations (AND, OR, NOT, XOR, NO-OP)
- 🔄 Gradient descent optimization for network training
- 📊 Interactive visualization of network structure and signal flow
- 📋 Test data validation with detailed metrics
- 🔍 Confusion matrix for binary classification problems

## Python Implementation 🐍

The Python implementation (`python-implementation/network.py`) provides the core functionality:

- Binary operation neurons: `And`, `Or`, `Not`, `Xor`, `NoOp`
- Network structure with configurable layers
- Training via gradient descent
- Network evaluation capabilities

### Example Usage

```python
# Create a network with 3 inputs and three layers of sizes 3, 2, and 2
network = Network(3, [3, 2, 2])

# Define a task (e.g., 3-bit binary addition)
def add_three_bits(input_values):
    sum = input_values[0] + input_values[1] + input_values[2]
    return [sum // 2, sum % 2]  # Returns [carry, sum]

# Train the network
GradientDescent().configure(
    network=network,
    evaluator=NetworkEvaluator().set_inputs_based_on_function(add_three_bits, 3),
    mix_up_coefficient=0.4
).run(max_steps=1000)
```

## Visualization Tool 🖥️

The visualization tool (`visualization/network.html`) provides an interactive interface to:

- Visualize network architecture
- Test inputs and observe signal propagation
- Validate network performance against test data
- View detailed metrics for binary classifiers

### How to Use the Visualization Tool

1. Open `visualization/network.html` in a web browser
2. Paste your network JSON configuration
3. Click "Visualize Network" to render the network
4. Set input values and click "Run Network" to see signals flow through the network
5. Use the test data section to validate network performance

## Getting Started 🚀

### Prerequisites

- Python 3.6+
- Web browser for visualization

### Setup

1. Clone this repository
2. Run examples in the Python implementation
3. Open the HTML file in a browser to use the visualization tool

## Applications 💡

This network has been successfully used for:
- Binary addition
- Cellular automata rules (Conway's Game of Life neighborhood evaluation)
- Other binary classification problems

## License 📝

[Your license information here]

## Acknowledgements 🙏

[Your acknowledgements here]
