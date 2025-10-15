import json
from network import Network
import pathlib
from evaluation import NetworkEvaluator
from main import three_two_bits_and_center, second_column_is_not_3

script_dir = pathlib.Path(__file__).parent.resolve()
json_file_name = 'best-network.json'
json_file_path = script_dir / json_file_name

with open(json_file_path, 'r') as f:
    json_data = f.read()


netw = Network.from_json(json_data).get_pruned()
loaded_as_json = json.dumps(netw.json())
print(loaded_as_json)

test_data = NetworkEvaluator() \
        .set_inputs_based_on_function(three_two_bits_and_center, 7) \
        .filter_inputs(second_column_is_not_3) \
        .test_data_json()['data']

csv_string = ''

for i in range(netw.input_size):
    csv_string += f"i_{i}, "

for i, layer in enumerate(netw.layers):
    for j, neuron in enumerate(layer.neurons):
        csv_string += f"n_lay-{i}_n-{j}, "

csv_string = csv_string[:-2] + '\n'

for inputs, expected in [(x['input'], x['expected_output']) for x in test_data]:

    output = netw.evaluate_get_values_of_all(inputs)
    # output look like this: [[0, 1, 0, 1, 0, 1, 1], [1, 0, 0, 1, 0, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0], [1, 1, 0], [1, 0], [0]]
    flatten_output = [item for sublist in output for item in sublist]

    csv_string += ', '.join([str(i) for i in inputs] + [str(x) for x in flatten_output]) + '\n'

# now I have something like this:
# i_0, i_1, i_2, i_3, i_4, i_5, i_6, n_0_0, n_0_1, n_0_2, n_0_3, n_0_4, n_0_5, n_0_6, n_0_7, n_0_8, n_1_0, n_1_1, n_1_2, n_1_3, n_1_4, n_1_5, n_2_0, n_2_1, n_2_2, n_3_0, n_3_1, n_4_0
# 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
# 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0
# 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0
# 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1
# ...

# i want all to find the same columns if there are any. use numpy:

import numpy as np
import pandas as pd
import io  # Add this import for StringIO

# Convert the CSV string to a pandas DataFrame
try:
    # Try with comma-space as separator
    df = pd.read_csv(io.StringIO(csv_string), sep=', ', engine='python')
except:
    # Try after cleaning up the CSV string
    csv_string_clean = csv_string.replace(', ', ',')
    df = pd.read_csv(io.StringIO(csv_string_clean))

# Find identical columns
column_groups = []
processed_columns = set()
columns = df.columns.tolist()

for i, col1 in enumerate(columns):
    if col1 in processed_columns:
        continue
    
    # Start a new group with this column
    current_group = [col1]
    processed_columns.add(col1)
    
    # Find all columns identical to col1
    for col2 in columns[i+1:]:
        if col2 not in processed_columns and df[col1].equals(df[col2]):
            current_group.append(col2)
            processed_columns.add(col2)
    
    # Only add groups with more than one column (duplicates found)
    if len(current_group) > 1:
        column_groups.append(current_group)

# Display the results
if column_groups:
    print("\nIdentical column groups found:")
    for i, group in enumerate(column_groups):
        print(f"Group {i+1}: {', '.join(group)}")
        
        # Identify neuron types in this group
        input_neurons = [col for col in group if col.startswith('i_')]
        hidden_neurons = [col for col in group if col.startswith('n_')]
        
        if input_neurons:
            print(f"  Input neurons: {', '.join(input_neurons)}")
        if hidden_neurons:
            print(f"  Hidden neurons: {', '.join(hidden_neurons)}")
            
        print()
else:
    print("No identical columns found.")

# Optionally, you could modify the network to remove redundant neurons here
# by updating the network structure based on the found identical columns


