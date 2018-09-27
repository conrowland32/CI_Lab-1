import sys
import random
from io_functions import read_training_data, build_layer
from training_functions import forward_pass, backprop

INPUT_DIMENSIONS = 3
HIDDEN_LAYER_DIMENSIONS = 11
OUTPUT_DIMENSIONS = 2
MOMENTUM = 0.3
LEARNING_RATE = 0.7


def main():
    sum_squared_errors = 0

    # Read in training data from cross_data
    training_data = read_training_data(
        './assets/cross_data (3 inputs - 2 outputs).csv', INPUT_DIMENSIONS, OUTPUT_DIMENSIONS)

    # Read and build hidden layer from w1 and b1
    hidden_layer = build_layer(
        './assets/w1 (3 inputs - 11 nodes).csv', './assets/b1 (11 nodes).csv')

    # Read and build output layer from w2 and b2
    output_layer = build_layer(
        './assets/w2 (from 11 to 2).csv', './assets/b2 (2 output nodes).csv')

    for neuron in hidden_layer:
        # print(neuron.weights, neuron.bias)
        for weight in neuron.weights:
            print(round(weight, 4), end=' ')
        print('\t/  ', round(neuron.bias, 4))
    for neuron in output_layer:
        # print(neuron.weights, neuron.bias)
        for weight in neuron.weights:
            print(round(weight, 4), end=' ')
        print('  /  ', round(neuron.bias, 4))
    print('\n')

    # Train network for one epoch
    for sample in training_data:
        sum_squared_errors += forward_pass(sample, hidden_layer, output_layer)
        backprop(sample, hidden_layer,
                 output_layer, MOMENTUM, LEARNING_RATE)

    for neuron in hidden_layer:
        # print(neuron.weights, neuron.bias)
        for weight in neuron.weights:
            print(round(weight, 4), end=' ')
        print('\t/  ', round(neuron.bias, 4))
    for neuron in output_layer:
        # print(neuron.weights, neuron.bias)
        for weight in neuron.weights:
            print(round(weight, 4), end=' ')
        print('  /  ', round(neuron.bias, 4))
    print('SSE:  ', sum_squared_errors / (2 * len(training_data)))


if __name__ == '__main__':
    main()
