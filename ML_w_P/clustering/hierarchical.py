import matplotlib.pyplot as plt 
import numpy as np 

x = np.array([[7, 8], [12, 20], [17, 19], [26, 15], [32, 37], [87, 75], [73, 85], [62, 80], [73, 60], [87, 96]])
labels = range(1, 11) 
plt.figure(figsize=(10, 7)) 
plt.subplots_adjust(bottom=0.1) 
plt.scatter(x[:, 0], x[:, 1], label = 'True Positions')

for label, x, y in zip(labels, x[:, 0], x[:, 1]):
    plt.annotate(label, xy = (x, y), xytext = (-3, 3), textcoords = 'offset points', ha = 'right', va = 'bottom') 
plt.show()


# from scipy.cluster.hierarchy import dendrogram, linkage 
# linked = linkage(x, 'single') 
# labelList = range(1, 11) 
# plt.figure(figsize = (10, 7)) 
# dendrogram(linked, orientation = 'top', labels = labelList, distance_sort = 'descending', show_leaf_counts = True) 
# plt.show()

from sklearn.cluster import AgglomerativeClustering 
cluster = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
cluster.fit_predict(x) 

plt.scatter(x[:, 0], x[:, 1], c= cluster.labels_, cmap = 'rainbow')
plt.show()