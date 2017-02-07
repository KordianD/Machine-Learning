import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def start():

    # Read data
    data = sp.genfromtxt("web_traffic.tsv", delimiter = '\t')

    # Split data
    x = data[:, 0]
    y = data[:, 1]

    # We need to clean data
    x = x[~sp.isnan(y)]
    y = y[~sp.isnan(y)]

    plot_data(x, y)

    # Add columns of ones
    b = np.ones((len(x), 1))
    x = x.reshape((len(x), 1))
    y = y.reshape((len(x), 1))
    x = np.concatenate((b, x), axis = 1)

    # Initial theta
    theta = np.zeros((2, 1))

    initial_cost = cost_function(theta, x, y)

    print("Initial cost ", initial_cost)

    theta = gradientDescent(theta, x, y, 0.000001,1000)
    print(theta)

    plt.plot(x[:, 1], x.dot(theta), 'b.')
    plt.show()

def cost_function(theta, x, y):
    """Calculate cost for given theta"""
    cost = sum((x.dot(theta) - y) ** 2)
    return cost

def plot_data(x, y):
    plt.scatter(x, y)
    plt.title("Web traffic over the last month")
    plt.xlabel("Time")
    plt.ylabel("Hits/hour")
    plt.grid()


def gradientDescent(theta, x, y, alpha, num_iter):
    m = len(y)

    for i in range(num_iter):
        theta = theta - ((1 / m) * ((x.dot(theta) - y).transpose()).dot(x)).transpose() * alpha
    return theta

start()