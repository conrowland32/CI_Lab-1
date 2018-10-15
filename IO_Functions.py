import csv
import random
from neuron import Neuron


def read_training_data(filename, input_dimensions, output_nodes):
    training_data = []

    # Build training data with inputs and desired outputs
    with open(filename, 'r') as input_file:
        reader = csv.reader(input_file)
        # Iterate CSV rows / data instances
        for row in reader:
            input = []
            # Iterate input values
            for i in range(0, input_dimensions):
                input.append(float(row[i]))
            # Iterate output values
            for i in range(0, output_nodes):
                input.append(int(row[i + input_dimensions]))
            training_data.append(input)

    return training_data


def build_layer(weights_filename, biases_filename, prev_layer_size):
    new_layer = []

    # Build neurons with starting weights
    with open(weights_filename, 'r') as weights_file:
        reader = csv.reader(weights_file)
        # Iterate CSV rows / neurons
        for row in reader:
            new_neuron = Neuron()
            weights = []
            # Iterate weights
            for i in range(prev_layer_size):
                weights.append(float(row[i]))
            new_neuron.set_weights(weights)
            new_layer.append(new_neuron)

    # Add starting biases
    with open(biases_filename, 'r') as biases_file:
        reader = csv.reader(biases_file)
        # Iterate CSV rows / neurons
        for index, row in enumerate(reader):
            for bias in row:
                new_layer[index].set_bias(float(bias))

    return new_layer


def read_txt_input(filename):
    training_data = []
    with open(filename, 'r') as input_file:
        lines = input_file.readlines()
        del lines[0]
        del lines[0]
        for line in lines:
            line = line.strip()
            clean_line = line.split(' ')
            clean_line = [x for x in clean_line if x != '']
            training_data.append(clean_line)
            for i, value in enumerate(clean_line):
                if i < 4:
                    clean_line[i] = float(value)
                elif int(value) == 1:
                    clean_line[i] = 0
                    clean_line.append(1)
                    break
                else:
                    clean_line[i] = 1
                    clean_line.append(0)
                    break
    return training_data


def build_custom_layer(prev_layer_size, layer_size):
    new_layer = []

    for _ in range(layer_size):
        new_neuron = Neuron()
        weights = []
        for _ in range(prev_layer_size):
            weights.append(random.random() * 2 - 1)
        new_neuron.set_weights(weights)
        new_neuron.set_bias(random.random() * 2 - 1)
        new_layer.append(new_neuron)
    return new_layer
