import json

class Function:
    def eval(self, input_values: list[int]) -> int:
        raise NotImplementedError("This method should be overridden by subclasses")
    
    @staticmethod
    def from_json(json_string: str) -> 'Function':
        data = json.loads(json_string)
        operation = data['operation']
        arguments_indexes = data['arguments_indexes']

        if operation == 'And':
            return And(arguments_indexes[0], arguments_indexes[1])
        elif operation == 'Or':
            return Or(arguments_indexes[0], arguments_indexes[1])
        elif operation == 'Not':
            return Not(arguments_indexes[0])
        elif operation == 'Xor':
            return Xor(arguments_indexes[0], arguments_indexes[1])
        elif operation == 'NoOp':
            return NoOp(arguments_indexes[0])
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
    def is_same(self, other: 'Function') -> bool:
        if not isinstance(other, Function):
            return False
        
        if self.__class__ != other.__class__:
            return False
        
        if isinstance(self, BinOp):
            return self.fst_neuron_idx == other.fst_neuron_idx and self.sec_neuron_idx == other.sec_neuron_idx
        
        if isinstance(self, UnOp):
            return self.fst_neuron_idx == other.fst_neuron_idx
        
        return True

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