import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


def main():
    X, Y = create_data()

    plt.scatter(X[:, 0], X[:, 1], c=Y, marker='o', s=70)
    plt.axis([-1, 6, -1, 6])

    kernel = {'rbf', 'sigmoid'}
    gamma = [1/100, 5, 10]
    Cp = [1.0, 5.0, 10.0]

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))


    for ker in kernel:
        for g in gamma:
            for coef in Cp:

                clf = SVC(kernel=ker, gamma=g, C=coef, probability=True)
                clf.fit(X, Y)
                Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)

                plt.figure()
                plt.contourf(xx, yy, Z, alpha=0.4)
                plt.scatter(X[:, 0], X[:, 1], c=Y, alpha=0.8)
                plt.title('Kernel:' + ker + '  Gamma:' + str(g) + '  C:' + str(coef))



    plt.show()


def create_data(num_points=100):

    x_cord1 = (np.random.uniform(0, 2.7, num_points)).reshape((num_points, 1))
    y_cord1 = np.random.uniform(0, 5, num_points).reshape((num_points, 1))

    x_cord2 = (5 - np.random.uniform(0, 2.7, num_points)).reshape((num_points, 1))
    y_cord2 = np.random.uniform(0, 5, num_points).reshape((num_points, 1))

    Y = [0] * num_points
    Y = np.array(Y)
    Y = np.concatenate((Y, np.array([1] * num_points)))

    X = np.concatenate((x_cord1, y_cord1), axis=1)
    temp = np.concatenate((x_cord2, y_cord2), axis=1)

    X = np.concatenate((X, temp), axis=0)

    return (X, Y)


main()