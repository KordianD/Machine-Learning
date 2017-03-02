import pandas
import numpy as np
import matplotlib.pyplot as plt

def prepare_data():
    X, Y = read_and_normalize_data()

    return split_training_and_test_set(X, Y)

def read_and_normalize_data():
    """Read and normalize data dividing by max value, return X, Y"""

    data = pandas.read_csv("classification.csv", header=None)

    first_column = np.array(data[0]).reshape((len(data[0]), 1))
    first_column = first_column / max(abs(first_column))

    second_column = np.array(data[1]).reshape((len(data[1]), 1))
    second_column = second_column / max(abs(second_column))

    X = np.concatenate((first_column, second_column), axis=1)
    Y = np.array(data[2]).reshape((len(data[2]), 1))

    return X, Y

def split_training_and_test_set(X, Y):
    """Split data for training and test (7/10)"""
    split = int(len(Y) / 10 * 7)

    X_train = X[:split, :]
    Y_train = Y[:split, :]

    X_test = X[split:, :]
    Y_test = Y[split:, :]

    return X_train, Y_train, X_test, Y_test

def show_data(X, Y):
    """Plot the data"""
    plt.scatter(X[:, 0], X[:, 1], s=60, c=Y, cmap=plt.cm.Spectral)
    plt.show()
