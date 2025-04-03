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