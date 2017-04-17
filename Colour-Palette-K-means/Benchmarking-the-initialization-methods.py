# Run on it a k-means clustering algorithm (with k=9) and the following initialisation methods:
# - fully random;
# - Forgy;
# - random partition;
# - k-means++.
# Silhouette - measure
# The silhouette value is a measure of how similar an object is to its own cluster (cohesion)
# compared to other clusters (separation). The silhouette ranges from -1 to 1,
# where a high value indicates that the object is well matched to its own cluster
# and poorly matched to neighboring clusters.

from sklearn.metrics import  silhouette_score
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# We create dataset which be easy to be clustered
data = np.array([[1, 1]])

x = np.arange(1, 6, 2)
y = np.arange(1, 6, 2)

number_of_points = 100
number_of_clusters = 9

for i in x:
    for j in y:
        temp = np.array([[np.random.uniform(i, i + 1), np.random.uniform(j, j + 1)] for x in range(number_of_points)])
        data = np.concatenate((data, temp), axis=0)

# Plot the dataset
plt.figure(1)
plt.scatter(data[:, 0], data[:, 1])


# K-means++ use and random
results = []
resultsR = []


for i in range(10):
    kmeans = KMeans(n_clusters=9, init='k-means++', n_init=1).fit_predict(data)
    kmeansR = KMeans(n_clusters=9, init='random', n_init=1).fit_predict(data)
    results.append(silhouette_score(data, kmeans))
    resultsR.append(silhouette_score(data, kmeansR))

# The Forgy method randomly chooses k observations from the data set and uses these as the initial means
# Firstly, we have to randomly selects some points from our data
resultsF = []
for i in range(10):
    randomIndexes = np.random.uniform(0, number_of_points*number_of_clusters, size=number_of_clusters)
    randomPoints = np.array([data[int(randomIndexes[i])] for i in range(len(randomIndexes))])
    kmeansF = KMeans(n_clusters=number_of_clusters, init=randomPoints, n_init=1).fit_predict(data)
    resultsF.append(silhouette_score(data, kmeansF))


# The Random Partition method first randomly assigns a cluster to each observation
# and then proceeds to the update step, thus computing the initial mean to be the centroid
# of the cluster's randomly assigned points

resultsRP = []


for i in range(10):
    cluster_labels = np.random.uniform(0, number_of_clusters, size=number_of_clusters*number_of_points)
    # Cast to int needed
    cluster_labels = np.array([int(cluster_labels[i]) for i in range(len(cluster_labels))])

    sum_of_coords = np.ones((number_of_clusters, 2))
    how_many_times = [0] * 9

    for j in range(len(cluster_labels)):
        sum_of_coords[cluster_labels[j]][0] += data[j][0]
        sum_of_coords[cluster_labels[j]][1] += data[j][1]
        how_many_times[cluster_labels[j]] += 1

    for j in range(number_of_clusters):
        sum_of_coords[j][0] /= how_many_times[j]
        sum_of_coords[j][1] /= how_many_times[j]

    kmeansRP = KMeans(n_clusters=9, init=sum_of_coords, n_init=1).fit_predict(data)
    resultsRP.append(silhouette_score(data, kmeansRP))


# Before we create plots, we have to calculate standard deviations to error bars

plt.figure(2)

plt.errorbar(range(10), results, yerr=np.std(results), fmt='*')
plt.axis([0, 10, 0, 1])
plt.title('Silhouette results - kMeans++')
plt.grid()


plt.figure(3)
plt.scatter(data[:,0], data[:,1], c=kmeans)
plt.title('How kMeans++ gathers clusters for our data')



plt.figure(4)
plt.errorbar(range(10), resultsR, yerr=np.std(resultsR), fmt='*')
plt.axis([0, 10, 0, 1])
plt.title('Silhouette results - random')
plt.grid()


plt.figure(5)
plt.scatter(data[:,0], data[:,1], c=kmeansR)
plt.title('How clusters look Random clustering')


plt.figure(6)
plt.errorbar(range(10), resultsF, yerr=np.std(resultsF), fmt='*')
plt.axis([0, 10, 0, 1])
plt.title('Silhouette results - Forgy')
plt.grid()

plt.figure(7)
plt.scatter(data[:,0], data[:,1], c=kmeansF)
plt.title('How clusters look Forgy clustering')


plt.figure(8)
plt.errorbar(range(10), resultsRP, yerr=np.std(resultsRP), fmt='*')
plt.axis([0, 10, 0, 1])
plt.title('Silhouette results - Random Partition')
plt.grid()

plt.figure(9)
plt.scatter(data[:,0], data[:,1], c=kmeansRP)
plt.title('How clusters look Random Partition')

plt.show()

