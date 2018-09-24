import sys
import random
from IO_Functions import read_training_data, build_hidden_layer

INPUT_DIMENSIONS = 3
HIDDEN_LAYER_DIMENSIONS = 11
OUTPUT_DIMENSIONS = 2


def main():
    # Read in training data from cross_data
    training_data = read_training_data(
        './assets/cross_data (3 inputs - 2 outputs).csv', INPUT_DIMENSIONS, OUTPUT_DIMENSIONS)

    # Read and build hidden layer from w1 and b1
    hidden_layer = build_hidden_layer(
        './assets/w1 (3 inputs - 11 nodes).csv', './assets/b1 (11 nodes).csv')


if __name__ == '__main__':
    main()
