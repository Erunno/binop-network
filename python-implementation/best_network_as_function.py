from main import three_two_bits_and_center, second_column_is_not_3
from evaluation import NetworkEvaluator

def gol_func(i1, i2, i3, i4, i5, i6, i7):
    # Hidden layer 1
    h1_0 = i2 | i6
    h1_1 = i7
    h1_2 = i1 & i4
    h1_3 = i3 | i4
    h1_4 = i1
    h1_5 = i2 & i6
    h1_6 = i3 ^ i5
    h1_7 = i1 ^ i3

    # Hidden layer 2
    h2_0 = h1_1
    h2_1 = h1_4 ^ h1_6
    h2_2 = h1_6 & h1_7
    h2_3 = h1_5
    h2_4 = h1_0 ^ h1_3
    h2_5 = h1_0 & h1_2

    # Hidden layer 3
    h3_0 = h2_3 | h2_5
    h3_1 = h2_0 | h2_1
    h3_2 = h2_2 ^ h2_4

    # Hidden layer 4
    h4_0 = h3_0 ^ h3_2
    h4_1 = h3_1 & h3_2

    # Output layer
    o1 = h4_0 & h4_1

    return o1

def main():
    data = NetworkEvaluator() \
        .set_inputs_based_on_function(three_two_bits_and_center, 7) \
        .filter_inputs(second_column_is_not_3) \
        .test_data_json()['data']
    
    all_inputs = [x['input'] for x in data]
    all_expected = [x['expected_output'] for x in data]

    for input, expected in zip(all_inputs, all_expected):
        i1, i2, i3, i4, i5, i6, i7 = input
        expected_output = expected[0]
        calculated_output = gol_func(i1, i2, i3, i4, i5, i6, i7)
        
        if expected_output != calculated_output:
            print(f"Mismatch: Input: {input}, Expected: {expected_output}, Calculated: {calculated_output}")

    print("All inputs processed.")

if __name__ == "__main__":
    main()