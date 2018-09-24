import sys
import math


class Neuron:
    def __init__(self, activation_function):
        if activation_function is 'sigmoid':
            def calc_y(self):
                return 1 / (1 + math.exp(-self.v))
        else:
            sys.exit(Exception('ERROR: Unknown activation function'))

    def set_weights(self, weights):
        self.weights = weights

    def set_bias(self, bias):
        self.bias = bias
