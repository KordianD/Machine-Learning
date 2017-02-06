import numpy as np
import math
import operator

data = np.genfromtxt("KNNdata.csv", delimiter = ',', skip_header = 1)
data = data[:,2:]

np.random.shuffle(data)
X = data[:, range(5)]
Y = data[:, 5]

def distance(instance1, instance2):
    dist = 0.0
    for i in range(len(instance1)):
        dist += pow((instance1[i] - instance2[i]), 2)
    return math.sqrt(dist)

# Calculating distances between all data, return sorted  k-elements list (whole element and output)
def getNeighbors(trainingSetX, trainingSetY, testInstance, k):
    distances = []
    for i in range(len(trainingSetX)):
        dist = distance(testInstance, trainingSetX[i])
        distances.append((trainingSetX[i], dist, trainingSetY[i]))
    distances.sort(key=operator.itemgetter(1))
    neighbour = []
    for elem in range(k):
        neighbour.append((distances[elem][0], distances[elem][2]))
    return neighbour

#return answer
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = int(neighbors[x][-1])
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse = True)
    return sortedVotes[0][0]

#return accuracy, your predicitons and actual values
def getAccuracy(testSetY, predictions):
    correct = 0
    for x in range(len(predictions)):
        if testSetY[x] == predictions[x]:
            correct += 1
    return (correct / (len(predictions))) * 100.0

def start():
    trainingSetX = X[:2000]
    trainingSetY = Y[:2000]
    testSetX = X[2000:]
    testSetY = Y[2000:]

    # generate predictions
    predictions = []
    k = 4
    for x in range(len(testSetX)):
        neighbors = getNeighbors(trainingSetX, trainingSetY, testSetX[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
    accuracy = getAccuracy(testSetY, predictions)
    print('Accuracy: ' + str(accuracy))

start()