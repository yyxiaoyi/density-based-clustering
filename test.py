# -*- coding: utf-8 -*-

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import distance
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from clustering import DBSCAN

sample_size = 1000
x, y = datasets.make_blobs(sample_size)
x = StandardScaler().fit_transform(x)

# run DBSCAN
Eps = 0.2
MinPts = 3
distance_matrix = distance.squareform(distance.pdist(x, 'minkowski'))

t0 = time.time()
labels, counts = DBSCAN(distance_matrix, Eps, MinPts)
t1 = time.time()

print '%.2fs.' % (t1-t0)

unique_labels = set(labels)
colors = [cm.Spectral(i) for i in np.linspace(0, 1, len(unique_labels))]
for i in range(sample_size):
    n_cluster = labels[i]
    if n_cluster == -1:
        color = [0, 0, 0, 1]
    else:
        color = colors[n_cluster]

    plt.plot(x[i][0], x[i][1], 'o', markerfacecolor=tuple(color),
             markeredgecolor='k', markersize=6)
plt.show()  


