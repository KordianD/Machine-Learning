'''We have a hypercube with edges length equal to 1.0. We randomly fill it with evenly distributed points.
What is the ratio between the standard deviation of distance between those points and the
average distance between them?'''

from random import randint
import math
import matplotlib.pyplot as plt
import numpy as np

length = 1
number_of_points = randint(100, 100)
max_dimension = 100
points = []

for i in range(max_dimension):
    points.append(np.random.uniform(-length / 2, length / 2, number_of_points))

all_result = []

# Calculate distance between all points
for i in range(number_of_points - 1):
    for j in range(i+1, number_of_points):
        dist = []
        dist_sqrt = []
        for k in range(max_dimension):
            if k == 0:
                dist.append((points[k][i] - points[k][j]) ** 2)
            else:
                dist.append((points[k][i] - points[k][j]) ** 2 + dist[k-1])
            dist_sqrt.append(math.sqrt(dist[k]))
        all_result.append(dist_sqrt)

all_results_dim = [0] * max_dimension

# Sum of all distances between points
for i in range(len(all_result)):
    for j in range(max_dimension):
        all_results_dim[j] += all_result[i][j]

# Mean of all distances
mean = [x / len(all_result) for x in all_results_dim]

# Standard deviation between all of distances
tempResult = [0] * max_dimension
for i in range(len(all_result)):
    for j in range(max_dimension):
        tempResult[j] += (all_result[i][j] - mean[j]) ** 2 / (len(all_result) - 1)

std = [math.sqrt(x) for x in tempResult]
print(std[0])

ratio = []
for i in range(len(std)):
    ratio.append(std[i]/mean[i])

#Plot data
plt.figure()
plt.plot(range(1, max_dimension + 1), ratio)
plt.xlabel('Number of dimensions')
plt.ylabel('Ratio')
plt.title('Standard deviation divided by mean distance between all points')
plt.xlim([1, max_dimension])
plt.show()







