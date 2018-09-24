import math


class Neuron:
    def set_weights(self, weights):
        self.weights = weights
        self.old_weights = weights

    def set_bias(self, bias):
        self.bias = bias
        self.old_bias = bias

    def calc_v(self, inputs, input_type):
        self.v = 0
        for index, weight in enumerate(self.weights):
            if input_type is 'inputs':
                self.v += weight * inputs[index]
            elif input_type is 'neurons':
                self.v += weight * inputs[index].y
        self.v += self.bias

    def calc_y(self):
        self.y = 1 / (1 + math.exp(-self.v))

    def calc_e(self, desired):
        self.e = desired - self.y

    def calc_delta(self, output_layer=None, index=None):
        if output_layer is None:
            self.delta = self.e * self.y * (1 - self.y)
        else:
            output_sum = 0
            for neuron in output_layer:
                output_sum += neuron.delta * neuron.old_weights[index]
            self.delta = output_sum * self.y * (1 - self.y)

    def calc_new_weights(self, previous_layer, momentum, learning_rate, layer_type):
        new_weights = [None] * len(self.weights)
        for index, weight in enumerate(self.weights):
            if layer_type is 'neurons':
                new_weights[index] = weight + \
                    (momentum * (weight - self.old_weights[index])) + \
                    (learning_rate * self.delta * previous_layer[index].y)
            elif layer_type is 'inputs':
                new_weights[index] = weight + \
                    (momentum * (weight - self.old_weights[index])) + \
                    (learning_rate * self.delta * previous_layer[index])
        self.old_weights = self.weights
        self.weights = new_weights

        new_bias = self.bias + \
            (momentum * (self.bias - self.old_bias)) + \
            (learning_rate * self.delta)
        self.old_bias = self.bias
        self.bias = new_bias
