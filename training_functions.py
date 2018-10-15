from neuron import Neuron


def forward_pass(sample, hidden_layer, output_layer, test=False):
    sum_squared_errors = 0

    # Calculate induced local field and output for each hidden neuron
    for neuron in hidden_layer:
        neuron.calc_v(sample, 'inputs')
        neuron.calc_y()

    # Calculate induced local field, output, and error for each output neuron
    for index, neuron in enumerate(output_layer):
        neuron.calc_v(hidden_layer, 'neurons')
        neuron.calc_y()
        if not test:
            neuron.calc_e(sample[len(sample) - len(output_layer) + index])
            sum_squared_errors += (neuron.e)**2
        else:
            return neuron.y

    # Return SSE for current sample
    return sum_squared_errors


def backprop(sample, hidden_layer, output_layer, momentum, learning_rate):
    # Calculate delta, new weights for each neuron
    for index, neuron in enumerate(output_layer):
        neuron.calc_delta()
        neuron.calc_new_weights(hidden_layer, momentum,
                                learning_rate, 'neurons')
    for index, neuron in enumerate(hidden_layer):
        neuron.calc_delta(output_layer, index)
        neuron.calc_new_weights(sample, momentum, learning_rate, 'inputs')


def test_forward_pass(sample, hidden_layer, output_layer):
    # Calculate induced local field and output for each hidden neuron
    for neuron in hidden_layer:
        neuron.calc_v(sample, 'inputs')
        neuron.calc_y()

    # Calculate induced local field, output, and error for each output neuron
    sample_outputs = []
    for neuron in output_layer:
        neuron.calc_v(hidden_layer, 'neurons')
        neuron.calc_y()
        sample_outputs.append(neuron.y)

    if sample_outputs.index(max(sample_outputs)) == 0:
        if sample[-1] == 1:
            return [0, 1]
        else:
            return [1, 1]
    else:
        if sample[-1] == 0:
            return [1, 0]
        else:
            return [0, 0]
