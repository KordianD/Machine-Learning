'''We have a hyperball with a radius equal to 1.0 inscribed inside a hypercube with edges length equal to 2.0.
Hyperball in multidimensional space is defined as a set of points whose distance from its centre is no greater
than its radius. We randomly fill the hypercube with evenly distributed points.
What % of those points would be inside the hyperball, and what % outside – in the corners?'''


from random import randint
import numpy as np
import math
import matplotlib.pyplot as plt

radius = 1.0
length = 2.0
max_dimension = 10
all_result = []

# 10 times we want to calculate this
for q in range(10):
    result = [0] * 10
    points = []

    # We want to randomly chose number of points
    number_of_points = randint(100, 300)

    # We want all points uniformly distributed
    for i in range(max_dimension):
        points.append(np.random.uniform(-length/2, length/2, number_of_points))

    # Calculate how many points are in hyperball
    for i in range(number_of_points):
        for j in range(max_dimension):
            distance = 0
            correct = 0
            for k in range(j + 1):
                distance += points[k][i] ** 2
            distance = math.sqrt(distance)
            if distance <= radius:
                result[j] += 1

    # Calculate correctness
    for i in range(10):
        result[i] = result[i] / number_of_points * 100

    # Add result
    all_result.append(result)


# We want to count std for every dimension independent one by one
sum = [0] * 10

for i in range(max_dimension):
    for j in range(max_dimension):
        sum[j] += all_result[i][j]

# Calculate mean
mean = [x / max_dimension for x in sum]

# Calculate standard deviation
tempResult = [0] * 10
for i in range(max_dimension):
    for j in range(max_dimension):
        tempResult[j] += (all_result[i][j] - mean[j]) ** 2 / (max_dimension - 1)

std = [math.sqrt(x) for x in tempResult]

#Plot data
plt.figure()
plt.errorbar(range(1, max_dimension + 1), mean, yerr=std)
plt.xlabel('Number of dimensions')
plt.ylabel('Percent of points in hyperball')
plt.ylim([0, 100])
plt.xlim([1, 10])
plt.show()




