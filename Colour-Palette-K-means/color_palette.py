import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from sklearn.metrics import  silhouette_score

def recreate_image(codebook, labels, w, h):
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image


# Load the Summer Palace photo
china = load_sample_image("china.jpg")


plt.figure()
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.imshow(china)


# Convert to range between 0-1
china = np.array(china, dtype=np.float64) / 255
x, y, z = china.shape

# Reshape to proper dimensions for computation
data = np.reshape(china, (x * y, z))
print(x * y)

# Take only a small subset
subset = shuffle(data, random_state=0)[:1000]
test_data = shuffle(data, random_state=0)[:1000]

silhouette_result = []

for i in range(2, 129, 2):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=1).fit(subset)
    labels = kmeans.predict(data)
    result_sil = kmeans.predict(test_data)
    silhouette_result.append(silhouette_score(test_data, result_sil))


plt.figure()
plt.plot(range(2, 129, 2), silhouette_result, '-*')
plt.show()
