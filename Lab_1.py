import sys
import random
import numpy
import itertools
import matplotlib.pyplot as pyplot
import matplotlib.lines as mlines
from io_functions import read_training_data, build_layer, read_txt_input, build_custom_layer
from training_functions import forward_pass, backprop, test_forward_pass


def main():
    input_dimensions = 3
    hidden_layer_dimensions = 11
    output_dimensions = 2
    momentum = 0.3
    learning_rate = 0.7
    sum_squared_errors = 0

    # Read in training data from cross_data
    training_data = read_training_data(
        './assets/cross_data (3 inputs - 2 outputs).csv', input_dimensions, output_dimensions)

    # Read and build hidden layer from w1 and b1
    hidden_layer = build_layer(
        './assets/w1 (3 inputs - 11 nodes).csv', './assets/b1 (11 nodes).csv', input_dimensions)

    # Read and build output layer from w2 and b2
    output_layer = build_layer(
        './assets/w2 (from 11 to 2).csv', './assets/b2 (2 output nodes).csv', hidden_layer_dimensions)

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
                     output_layer, momentum, learning_rate)
        newError = sum_squared_errors / (2 * len(training_data))
        errors.append(newError)
        print('SSE:  ', newError)
        if lastError is not None and lastError - newError < 0.001:
            break
        lastError = newError
        random.shuffle(training_data)
    print('\n-----\n')

    # Plot sum of squared errors
    pyplot.plot(numpy.arange(0, len(errors), 1), errors, 'b.-')
    pyplot.title('3-Feature Training Errors')
    pyplot.xlabel('Training Epoch')
    pyplot.ylabel('Sum of Squared Errors')
    pyplot.grid()
    pyplot.savefig('test_results/3-feature_sse.png')
    pyplot.clf()

    # Sample [-2.1, 2.1] x [-2.1, 2.1] square and plot
    square_x = []
    square_y = []
    square_z = []
    square_colors = []
    x = -2.1
    while x <= 2.1:
        y = -2.1
        while y <= 2.1:
            square_x.append(x)
            square_y.append(y)
            square_z.append((random.random() - 0.5) / 20)
            y = round(y+0.01, 2)
        x = round(x+0.01, 2)

    for i in range(len(square_x)):
        sample = [square_x[i], square_y[i], square_z[i]]
        square_colors.append(forward_pass(
            sample, hidden_layer, output_layer, True))

    pyplot.scatter(square_x, square_y, c=square_colors, alpha=0.1)
    pyplot.title('3-Feature Classifications')
    pyplot.xlabel('Feature 1')
    pyplot.ylabel('Feature 2')
    pyplot.savefig('test_results/3-feature_classifications.png')
    pyplot.clf()

    # Remove 3rd feature and retrain
    for sample in training_data:
        del sample[2]

    hidden_layer = build_layer(
        './assets/w1 (3 inputs - 11 nodes).csv', './assets/b1 (11 nodes).csv', 2)

    output_layer = build_layer(
        './assets/w2 (from 11 to 2).csv', './assets/b2 (2 output nodes).csv', hidden_layer_dimensions)

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
                     output_layer, momentum, learning_rate)
        newError = sum_squared_errors / (2 * len(training_data))
        errors.append(newError)
        print('SSE:  ', newError)
        if lastError is not None and lastError - newError < 0.001:
            break
        lastError = newError
        random.shuffle(training_data)
    print('\n-----\n')

    # Plot sum of squared errors
    pyplot.plot(numpy.arange(0, len(errors), 1), errors, 'b.-')
    pyplot.title('2-Feature Training Errors')
    pyplot.xlabel('Training Epoch')
    pyplot.ylabel('Sum of Squared Errors')
    pyplot.grid()
    pyplot.savefig('test_results/2-feature_sse.png')
    pyplot.clf()

    # Sample [-2.1, 2.1] x [-2.1, 2.1] square and plot
    square_x = []
    square_y = []
    square_z = []
    square_colors = []
    x = -2.1
    while x <= 2.1:
        y = -2.1
        while y <= 2.1:
            square_x.append(x)
            square_y.append(y)
            square_z.append((random.random() - 0.5) / 20)
            y = round(y+0.01, 2)
        x = round(x+0.01, 2)

    for i in range(len(square_x)):
        sample = [square_x[i], square_y[i], square_z[i]]
        square_colors.append(forward_pass(
            sample, hidden_layer, output_layer, True))

    pyplot.scatter(square_x, square_y, c=square_colors, alpha=0.1)
    pyplot.title('2-Feature Classifications')
    pyplot.xlabel('Feature 1')
    pyplot.ylabel('Feature 2')
    pyplot.savefig('test_results/2-feature_classifications.png')
    pyplot.clf()

    # 2: Test different learning rates
    learning_rates = [0.01, 0.2, 0.7, 0.9]
    for epochs in [25, 250]:
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
            while len(errors) <= epochs:
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

        pyplot.title('Effect of Learning Rate on Training Error')
        pyplot.xlabel('Training Epoch')
        pyplot.ylabel('Sum of Squared Errors')
        pyplot.legend(handles=[red_line, green_line, blue_line, yellow_line])
        pyplot.grid()
        pyplot.savefig('test_results/learning_rates(' +
                       str(epochs) + ' epochs).png')
        pyplot.clf()

    # 3: Change momentum
    learning_rate = 0.01
    momentums = [0, 0.6]
    for epochs in [100, 500]:
        for momentum in momentums:
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
            while True:
                # One epoch
                sum_squared_errors = 0
                for sample in training_data:
                    sum_squared_errors += forward_pass(sample,
                                                       hidden_layer, output_layer)
                    backprop(sample, hidden_layer,
                             output_layer, momentum, learning_rate)
                newError = sum_squared_errors / (2 * len(training_data))
                errors.append(newError)
                print('SSE:  ', newError)
                if len(errors) > epochs:
                    break
                random.shuffle(training_data)
            print('\n-----\n')

            # Plot sum of squared errors
            if momentum == 0:
                pyplot.plot(numpy.arange(0, len(errors), 1), errors, 'r.-')
                red_line = mlines.Line2D(
                    [], [], color='red', marker='.', label='0')
            if momentum == 0.6:
                pyplot.plot(numpy.arange(0, len(errors), 1), errors, 'g.-')
                green_line = mlines.Line2D(
                    [], [], color='green', marker='.', label='0.6')

        pyplot.title(
            'Effect of Momentum on Training Error (' + str(epochs) + ' epochs)')
        pyplot.xlabel('Training Epoch')
        pyplot.ylabel('Sum of Squared Errors')
        pyplot.ylim(0.0)
        pyplot.legend(handles=[red_line, green_line])
        pyplot.grid()
        pyplot.savefig('test_results/momentums(' +
                       str(epochs) + ' epochs).png')
        pyplot.clf()

    # 4: Cross-Validation Tests
    input_dimensions = 4
    output_dimensions = 2
    learning_rate = 0.001
    momentum = 0.2

    training_data = read_txt_input('./assets/Two_Class_FourDGaussians500.txt')
    random.shuffle(training_data)
    size = len(training_data) / 5.0
    split_training_data = []
    running = 0.0

    while running < len(training_data):
        split_training_data.append(
            training_data[int(running):int(running + size)])
        running += size

    hidden_sizes = [4, 8, 16]
    for size in hidden_sizes:
        # Train network
        total_matrix = numpy.array([[0, 0], [0, 0]])
        average_errors = [0] * 50
        for fold in range(5):
            # Build a new hidden layer with random weights and biases
            hidden_layer = build_custom_layer(
                input_dimensions, size)

            # Build a new output layer with random weights and biases
            output_layer = build_custom_layer(
                size, output_dimensions)

            if fold == 0:
                training_data = split_training_data[1] + split_training_data[2] + \
                    split_training_data[3] + split_training_data[4]
            elif fold == 1:
                training_data = split_training_data[0] + split_training_data[2] + \
                    split_training_data[3] + split_training_data[4]
            elif fold == 2:
                training_data = split_training_data[0] + split_training_data[1] + \
                    split_training_data[3] + split_training_data[4]
            elif fold == 3:
                training_data = split_training_data[0] + split_training_data[1] + \
                    split_training_data[2] + split_training_data[4]
            else:
                training_data = split_training_data[0] + split_training_data[1] + \
                    split_training_data[2] + split_training_data[3]

            errors = []
            for _ in range(50):
                # One epoch
                sum_squared_errors = 0
                random.shuffle(training_data)
                for sample in training_data:
                    sum_squared_errors += forward_pass(sample,
                                                       hidden_layer, output_layer)
                    backprop(sample, hidden_layer, output_layer,
                             momentum, learning_rate)
                new_error = sum_squared_errors / (2 * len(training_data))
                errors.append(new_error)

            for i, error in enumerate(average_errors):
                updated_error = (error * fold + errors[i]) / (fold + 1)
                average_errors[i] = updated_error

            num_right = 0
            conf_matrix = numpy.array([[0, 0], [0, 0]])
            for sample in split_training_data[fold]:
                test_result = test_forward_pass(
                    sample, hidden_layer, output_layer)
                conf_matrix[test_result[0]][test_result[1]] += 1
                if test_result[0] == test_result[1]:
                    num_right += 1
            print(str(num_right) + ' / 200\t' + str(num_right/2) + '%')

            for i in range(2):
                for j in range(2):
                    total_matrix[i][j] += conf_matrix[i][j]

            if size == 8:
                plot_matrix(conf_matrix, 'Normalized Confusion Matrix: Fold ' + str(fold+1),
                            './test_results/cross_validation/conf_matrix_fold_' + str(fold+1))

        # Plot sum of squared errors
        pyplot.plot(numpy.arange(0, len(average_errors), 1),
                    average_errors, 'b.-')
        pyplot.title('Training Errors: ' +
                     str(size) + ' Hidden Neurons')
        pyplot.xlabel('Training Epoch')
        pyplot.ylabel('Sum of Squared Errors')
        pyplot.ylim(0, 0.4)
        pyplot.grid()
        pyplot.savefig('test_results/cross_validation/' +
                       str(size) + '-neuron_sse')
        pyplot.clf()

        plot_matrix(total_matrix, 'Normalized Confusion Matrix: ' + str(size) + ' Hidden Neurons',
                    './test_results/cross_validation/conf_matrix_' + str(size) + '_units')
        print('\n-----\n')


def plot_matrix(conf_matrix, title, output_filename):
    # Build confusion matrix
    conf_matrix = conf_matrix.astype(
        'float') / (conf_matrix.sum(axis=1)[:, numpy.newaxis])
    color_map = pyplot.cm.Blues
    pyplot.imshow(conf_matrix, interpolation='nearest',
                  cmap=color_map, vmin=0.0, vmax=1.0)
    pyplot.colorbar()
    tick_marks = numpy.arange(2)
    pyplot.xticks(tick_marks, [0, 1])
    pyplot.yticks(tick_marks, [0, 1])
    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        pyplot.text(j, i, format(conf_matrix[i, j], '.2f'), horizontalalignment='center',
                    color='white' if conf_matrix[i, j] > thresh else 'black')

    pyplot.title(title)
    pyplot.ylabel('Actual Class')
    pyplot.xlabel('Predicted Class')
    pyplot.savefig(output_filename)
    pyplot.clf()


if __name__ == '__main__':
    main()
