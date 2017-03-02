import numpy as np

class Neural_Network(object):
    def __init__(self):
        """Setting the structure"""
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 4

        # Randomly initialize wages
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self, X):
        """Forward propagation"""
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        predict = self.sigmoid(self.z3)
        return predict

    def sigmoid(self, z):
        """ Element wise sigmoid function"""
        return 1 / (1 + np.exp(-z))

    def sigmoidPrime(self, z):
        """ Derative of sigmoid function"""
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

    def costFunction(self, X, y):
        """Compute cost for given data and wages"""
        predict = self.forward(X)
        cost = 0.5 * sum((y - predict) ** 2)
        return cost

    def compute_gradient(self, X, y):
        """ Compute derivative for wages"""
        predict = self.forward(X)

        delta3 = np.multiply(-(y - predict), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T) * self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2

    def learn(self, X, y, learning_rate):
        """ Compute gradient and subtract from actual parameters"""
        dJdW1, dJdW2 = self.compute_gradient(X, y)

        self.W1 = self.W1 - learning_rate * dJdW1
        self.W2 = self.W2 - learning_rate * dJdW2

    @staticmethod
    def make_binary_output(predict):
        for i in range(len(predict)):
            if predict[i] >= 0.5:
                predict[i] = 1
            else:
                predict[i] = 0

        return predict

    @staticmethod
    def compute_accuracy(predict, y):
        accurate = 0
        for i in range(len(y)):
            if y[i] == predict[i]:
                accurate += 1

        return accurate / len(y)

