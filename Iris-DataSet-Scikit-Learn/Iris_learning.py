from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def main():
    iris = load_iris()

    x = iris.data
    y = iris.target

    [x, y] = shuffle_data(x, y)
    [x_train, y_train, x_test, y_test] = split_data(x, y)

    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(x_train, y_train)

    correctness = accuracy(x_test, y_test, knn)
    print(correctness)

def accuracy(x, y, knn):
    correct = 0
    for i in range(len(y)):
        if knn.predict(x[i].reshape(1,4)) == y[i]:
            correct += 1

    return correct/len(y)

def shuffle_data(x,y):
    y_vector = y.reshape((y.size, 1))
    merged = np.concatenate((x, y_vector), axis=1)

    np.random.shuffle(merged)

    x = merged[:, range(4)]
    y = merged[:, 4]
    return [x, y]

def split_data(x, y):
    n = len(y)

    x_train = np.array(x[:(n * 3) // 4, :])
    y_train = np.array(y[:(n * 3) // 4])

    x_test = np.array(x[(n * 3) // 4:, :])
    y_test = np.array(y[(n * 3) // 4:])

    return [x_train, y_train, x_test, y_test]

main()

