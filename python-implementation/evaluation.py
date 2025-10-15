from itertools import product
from network import Network

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
    
    @staticmethod
    def mcc(predicted_outputs, expected_outputs):
        """Calculate Matthews correlation coefficient"""
        predicted, expected = Metrics._check_binary_output(predicted_outputs, expected_outputs)
        
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        
        for prediction, expected_value in zip(predicted, expected):
            for pred_single_val, expected_single_val in zip(prediction, expected_value):
                if pred_single_val == 1 and expected_single_val == 1:
                    true_positives += 1
                elif pred_single_val == 0 and expected_single_val == 0:
                    true_negatives += 1
                elif pred_single_val == 1 and expected_single_val == 0:
                    false_positives += 1
                elif pred_single_val == 0 and expected_single_val == 1:
                    false_negatives += 1

        numerator = (true_positives * true_negatives) - (false_positives * false_negatives)
        denominator = ((true_positives + false_positives) * (true_positives + false_negatives) * 
                       (true_negatives + false_positives) * (true_negatives + false_negatives)) ** 0.5
        
        return numerator / denominator if denominator > 0 else 0

class NetworkEvaluator:
    def __init__(self):
        self.network = None
        self.input_values = []
        self.expected_outputs = []
        # self.evaluation_metric = Metrics.accuracy
        self.evaluation_metric = Metrics.mcc
        self.statistics_metrics = {
            'mcc': Metrics.mcc,
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
    
    def filter_inputs(self, filter_function: callable):
        self.expected_outputs = [self.expected_outputs[i] for i in range(len(self.expected_outputs)) if filter_function(self.input_values[i])]
        self.input_values = [inputs for inputs in self.input_values if filter_function(inputs)]
        return self

    def set_evaluation_metric(self, metric_function: str):
        if isinstance(metric_function, str):
            metric_function = getattr(Metrics, metric_function, None)
            
            if metric_function is None:
                raise ValueError(f"Invalid metric name: {metric_function}")
            
            return self
        
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

