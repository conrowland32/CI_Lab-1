import sys
import random
import numpy
import matplotlib.pyplot as pyplot
import matplotlib.lines as mlines
from io_functions import read_training_data, build_layer
from training_functions import forward_pass, backprop


def main():
    input_dimensions = 3
    hidden_layer_dimensions = 11
    output_dimensions = 2
    momentum = 0.3
    learning_rate = 0.7
    sum_squared_errors = 0

    # # Read in training data from cross_data
    # training_data = read_training_data(
    #     './assets/cross_data (3 inputs - 2 outputs).csv', input_dimensions, output_dimensions)

    # # Read and build hidden layer from w1 and b1
    # hidden_layer = build_layer(
    #     './assets/w1 (3 inputs - 11 nodes).csv', './assets/b1 (11 nodes).csv', input_dimensions)

    # # Read and build output layer from w2 and b2
    # output_layer = build_layer(
    #     './assets/w2 (from 11 to 2).csv', './assets/b2 (2 output nodes).csv', hidden_layer_dimensions)

    # # Train network
    # errors = []
    # lastError = None
    # while True:
    #     # One epoch
    #     sum_squared_errors = 0
    #     for sample in training_data:
    #         sum_squared_errors += forward_pass(sample,
    #                                            hidden_layer, output_layer)
    #         backprop(sample, hidden_layer,
    #                  output_layer, momentum, learning_rate)
    #     newError = sum_squared_errors / (2 * len(training_data))
    #     errors.append(newError)
    #     print('SSE:  ', newError)
    #     if lastError is not None and lastError - newError < 0.001:
    #         break
    #     lastError = newError
    #     random.shuffle(training_data)
    # print('\n-----\n')

    # # Plot sum of squared errors
    # pyplot.plot(numpy.arange(0, len(errors), 1), errors, 'b.-')
    # pyplot.xlabel('Training Epoch')
    # pyplot.ylabel('Sum of Squared Errors')
    # pyplot.savefig('test_results/3-feature_sse.png')
    # pyplot.clf()

    # # Sample [-2.1, 2.1] x [-2.1, 2.1] square and plot
    # square_x = []
    # square_y = []
    # square_z = []
    # square_colors = []
    # x = -2.1
    # while x <= 2.1:
    #     y = -2.1
    #     while y <= 2.1:
    #         square_x.append(x)
    #         square_y.append(y)
    #         square_z.append((random.random() - 0.5) / 20)
    #         y = round(y+0.01, 2)
    #     x = round(x+0.01, 2)

    # for i in range(len(square_x)):
    #     sample = [square_x[i], square_y[i], square_z[i]]
    #     square_colors.append(forward_pass(
    #         sample, hidden_layer, output_layer, True))

    # pyplot.scatter(square_x, square_y, c=square_colors, alpha=0.1)
    # pyplot.xlabel('Feature 1')
    # pyplot.ylabel('Feature 2')
    # pyplot.savefig('test_results/3-feature_classifications.png')
    # pyplot.clf()

    # # Remove 3rd feature and retrain
    # for sample in training_data:
    #     del sample[2]

    # hidden_layer = build_layer(
    #     './assets/w1 (3 inputs - 11 nodes).csv', './assets/b1 (11 nodes).csv', 2)

    # output_layer = build_layer(
    #     './assets/w2 (from 11 to 2).csv', './assets/b2 (2 output nodes).csv', hidden_layer_dimensions)

    # # Train network
    # errors = []
    # lastError = None
    # while True:
    #     # One epoch
    #     sum_squared_errors = 0
    #     for sample in training_data:
    #         sum_squared_errors += forward_pass(sample,
    #                                            hidden_layer, output_layer)
    #         backprop(sample, hidden_layer,
    #                  output_layer, momentum, learning_rate)
    #     newError = sum_squared_errors / (2 * len(training_data))
    #     errors.append(newError)
    #     print('SSE:  ', newError)
    #     if lastError is not None and lastError - newError < 0.001:
    #         break
    #     lastError = newError
    #     random.shuffle(training_data)
    # print('\n-----\n')

    # # Plot sum of squared errors
    # pyplot.plot(numpy.arange(0, len(errors), 1), errors, 'b.-')
    # pyplot.xlabel('Training Epoch')
    # pyplot.ylabel('Sum of Squared Errors')
    # pyplot.savefig('test_results/2-feature_sse.png')
    # pyplot.clf()

    # # Sample [-2.1, 2.1] x [-2.1, 2.1] square and plot
    # square_x = []
    # square_y = []
    # square_z = []
    # square_colors = []
    # x = -2.1
    # while x <= 2.1:
    #     y = -2.1
    #     while y <= 2.1:
    #         square_x.append(x)
    #         square_y.append(y)
    #         square_z.append((random.random() - 0.5) / 20)
    #         y = round(y+0.01, 2)
    #     x = round(x+0.01, 2)

    # for i in range(len(square_x)):
    #     sample = [square_x[i], square_y[i], square_z[i]]
    #     square_colors.append(forward_pass(
    #         sample, hidden_layer, output_layer, True))

    # pyplot.scatter(square_x, square_y, c=square_colors, alpha=0.1)
    # pyplot.xlabel('Feature 1')
    # pyplot.ylabel('Feature 2')
    # pyplot.savefig('test_results/2-feature_classifications.png')
    # pyplot.clf()

    # 2: Test different learning rates
    learning_rates = [0.01, 0.2, 0.7, 0.9]
    for rate in learning_rates:
        # Read in training data from cross_data
        training_data = read_training_data(
            './assets/cross_data (3 inputs - 2 outputs).csv', input_dimensions, output_dimensions)

        # Read and build hidden layer from w1 and b1
        hidden_layer = build_layer(
            './assets/w1 (3 inputs - 11 nodes).csv', './assets/b1 (11 nodes).csv', input_dimensions)

        # Read and build output layer from w2 and b2
        output_layer = build_layer('./assets/w2 (from 11 to 2).csv',
                                   './assets/b2 (2 output nodes).csv', hidden_layer_dimensions)

        # Train network
        errors = []
        lastError = None
        while True:
            # One epoch
            sum_squared_errors = 0
            for sample in training_data:
                sum_squared_errors += forward_pass(sample,
                                                   hidden_layer, output_layer)
                backprop(sample, hidden_layer,
                         output_layer, momentum, rate)
            newError = sum_squared_errors / (2 * len(training_data))
            errors.append(newError)
            print('SSE:  ', newError)
            if lastError is not None and lastError - newError < 0.001:
                if rate != 0.01 or len(errors) > 50:
                    break
            lastError = newError
            random.shuffle(training_data)
        print('\n-----\n')

        # Plot sum of squared errors
        if rate == 0.01:
            pyplot.plot(numpy.arange(0, len(errors), 1), errors, 'r.-')
            red_line = mlines.Line2D(
                [], [], color='red', marker='.', label='0.01')
        if rate == 0.2:
            pyplot.plot(numpy.arange(0, len(errors), 1), errors, 'g.-')
            green_line = mlines.Line2D(
                [], [], color='green', marker='.', label='0.2')
        if rate == 0.7:
            pyplot.plot(numpy.arange(0, len(errors), 1), errors, 'b.-')
            blue_line = mlines.Line2D(
                [], [], color='blue', marker='.', label='0.7')
        if rate == 0.9:
            pyplot.plot(numpy.arange(0, len(errors), 1), errors, 'y.-')
            yellow_line = mlines.Line2D(
                [], [], color='yellow', marker='.', label='0.9')

    pyplot.xlabel('Training Epoch')
    pyplot.ylabel('Sum of Squared Errors')
    pyplot.legend(handles=[red_line, green_line, blue_line, yellow_line])
    pyplot.savefig('test_results/learning_rates.png')
    pyplot.clf()


if __name__ == '__main__':
    main()
