# Simple implementation of bagging

import numpy as np
import pandas
from sklearn import neighbors
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# We read the data from website (about blood data)
data = pandas.read_csv("Blood.csv", header=None, delimiter=',')
N, M = data.shape
data = np.array(data)

training_size = int(N * 7/10)

X = data[:, :M-1]
Y = data[:, M-1]

X_train = X[:training_size]
X_test = X[training_size:]

Y_train = Y[:training_size]
Y_test = Y[training_size:]

# Create 2 KNN and 2 SVM classifiers
classifiers = []
classifiers.append(neighbors.KNeighborsClassifier(n_neighbors=3, metric='euclidean'))
classifiers.append(neighbors.KNeighborsClassifier(n_neighbors=5, metric='cityblock'))
classifiers.append(SVC(kernel='rbf', gamma=1, C=1, probability=True))
classifiers.append(SVC(kernel='sigmoid', gamma=5, C=5, probability=True))

classifiers2 = []
classifiers2.append(neighbors.KNeighborsClassifier(n_neighbors=3, metric='euclidean'))
classifiers2.append(neighbors.KNeighborsClassifier(n_neighbors=5, metric='cityblock'))
classifiers2.append(SVC(kernel='rbf', gamma=1, C=1, probability=True))
classifiers2.append(SVC(kernel='sigmoid', gamma=5, C=5, probability=True))

number_of_classifiers = 4
number_of_rep = 10
bagging = [0] * number_of_rep
independent = [0] * number_of_classifiers

for q in range(number_of_rep):

    # We will use 4 classifier (2 KNN and 2 SVM) with different parameters
    # We have to randomly choose 10% of the train test for 4 our classifiers
    indexes = []
    for i in range(number_of_classifiers):
        indexes.append(np.array([int(np.random.uniform(0, training_size)) for x in range(training_size//10)]))

    X_selected = []
    Y_selected = []

    # Add all randomly selected indexes to one list
    for i in range(number_of_classifiers):
        X_selected.append(np.array(X_train[indexes[i]]))
        Y_selected.append(np.array(Y_train[indexes[i]]))


    # Train models
    for i in range(number_of_classifiers):
        classifiers[i].fit(X_selected[i], Y_selected[i])

    # We want to predict and count votes from all classifiers
    results = [0] * len(Y_test)
    for i in range(len(Y_test)):
        for j in range(number_of_classifiers):
            results[i] += classifiers[j].predict(X_test[i].reshape(1, -1))

    # We want numpy array
    results = np.array(results)

    # We have to divide by 4 (because 4 classifiers)
    results = results / number_of_classifiers
    results = [0 if x < 2 else 1 for x in results]

    # We have to check correctness of our predictions
    correct = 0
    for i in range(len(Y_test)):
        if results[i] == Y_test[i]:
            correct += 1

    # Percentage rate
    correct /= len(Y_test)
    bagging[q] = correct

    # We get some prediction and acuration rate from our bagging technique
    # We will have to create 4 models independently
    # We choose the same range of parameters

    # Train models
    for i in range(number_of_classifiers):
        classifiers2[i].fit(X_test, Y_test)

# We will calculate correctness for each classifier independently
for i in range(len(Y_test)):
    for j in range(number_of_classifiers):
        temp = classifiers2[j].predict(X_test[i].reshape(1, -1))
        if temp == Y_test[i]:
            independent[j] += 1

# Plot out bagging results
bagging = np.array(bagging)
bagging_std = np.std(bagging)

plt.figure()
plt.errorbar(range(1, number_of_rep + 1), bagging, yerr=bagging_std)
plt.xlabel('Number of iteration')
plt.ylabel('Percent of correctness')
plt.title('Bagging techique')
plt.ylim([0, 1])
plt.xlim([1, 10])

# We will do the same for out classifier
independent = np.array(independent)
independent = independent / len(Y_test)

plt.figure()
plt.plot(range(1, number_of_classifiers + 1), independent, 'o')
plt.axis([0, 5, 0, 1])
plt.title('1.KNN=3, Euclidean 2.KNN=5, Cityblock\n 3. SVM - rbf 4. SVM - sigmoid')
plt.xlabel('Different classifier')
plt.ylabel('Percentage of correctness')
plt.show()
