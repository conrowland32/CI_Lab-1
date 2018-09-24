import csv
from Neuron import Neuron


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


def build_hidden_layer(weights_filename, biases_filename):
    hidden_layer = []

    # Build neurons with starting weights
    with open(weights_filename, 'r') as weights_file:
        reader = csv.reader(weights_file)
        # Iterate CSV rows / neurons
        for row in reader:
            new_neuron = Neuron('sigmoid')
            weights = []
            # Iterate weights
            for weight in row:
                weights.append(float(weight))
            new_neuron.set_weights(weights)
            hidden_layer.append(new_neuron)

    # Add starting biases
    with open(biases_filename, 'r') as biases_file:
        reader = csv.reader(biases_file)
        # Iterate CSV rows / neurons
        for index, row in enumerate(reader):
            for bias in row:
                hidden_layer[index].set_bias(float(bias))

    return hidden_layer
