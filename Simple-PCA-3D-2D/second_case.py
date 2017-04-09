from sklearn.decomposition import KernelPCA
from sklearn import decomposition
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab
import numpy as np

number_of_points = 400

# Out first dimension
xa = np.random.uniform(0, 400, 400)

# Our second dimension
xb = [i/20 + np.random.uniform(0, 1, 1) for i in xa[:200]]
xb2 = [-i/20 + 20 + np.random.uniform(0, 1, 1) for i in xa[200:]]

# Our added 3rd dimension
x3d = np.random.uniform(0, 1, 400).reshape((400, 1))

# We want to use numpy array for calculations
xb = np.array(xb)
xb2 = np.array(xb2)

xb = np.concatenate((xb, xb2), axis=0)

xa = np.reshape(xa, (number_of_points, 1))
xb = np.reshape(xb, (number_of_points, 1))

X = np.concatenate((xa, xb), axis=1)
X = np.concatenate((X, x3d), axis=1)

y = np.array([0] * 200).reshape((200, 1))
y1 = np.array([1] * 200).reshape((200, 1))
y = np.concatenate((y, y1), axis=0)

fig = pylab.figure()
ax = Axes3D(fig)

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)
plt.title('3-D points initial values')

plt.figure()

pca = decomposition.PCA(n_components=2)
pca.fit(X)
results = pca.transform(X)

plt.scatter(results[:, 0], results[:, 1], c=y)
plt.title('After PCA for 2-D')

## KERNEL

plt.figure()

pca_cosine = KernelPCA(n_components=2, kernel='cosine')
results_cosine = pca_cosine.fit_transform(X)

plt.scatter(results_cosine[:, 0], results_cosine[:, 1], c=y)
plt.title('After Kernel PCA Cosine for 2-D')

## SIGMOID

plt.figure()

pca_sigmoid = KernelPCA(kernel='sigmoid', coef0=0.04)
results_sigmoid = pca_sigmoid.fit_transform(X)

plt.scatter(results_sigmoid[:, 0], results_sigmoid[:, 1], c=y)
plt.title('After Kernel PCA Sigmoid for 2-D')

# RBF

plt.figure()

pca_rbf = KernelPCA(kernel='rbf', gamma=5)
results_rbf = pca_rbf.fit_transform(X)

plt.scatter(results_rbf[:, 0], results_rbf[:, 1], c=y)
plt.title('After Kernel PCA RBF for 2-D')
plt.show()




