from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_circles
from sklearn import decomposition
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab
import numpy as np

# Firstly, we want to create random dataset
# One looking like 2 circles

X, y = make_circles(n_samples=400, factor=.3, noise=.02)

column = np.random.uniform(0, 1, 400)
column = np.reshape(column, (400, 1))

X = np.concatenate((X, column), axis=1)

# We want to create 3-D figure
fig = pylab.figure()
ax = Axes3D(fig)

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)
plt.title('3-D points (circles) initial values')

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



