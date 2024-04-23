import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np 
from pandas import read_csv 
path = "/workspaces/Agiat_Ikazinat/machine_learning/ML_w_P/pima-indians-diabetes.csv"
headernames = ['preg', 'plag', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names = headernames) 
array = data.values 
x = array[:, 0:8]
y = array[:, 8]
print(data.shape)
print(data.head())

patient_data = data.iloc[:, 3:5].values 
import scipy.cluster.hierarchy as shc 
plt.figure(figsize= (10, 7)) 
plt.title("Patient Dendrograms") 
dend  = shc.dendrogram(shc.linkage(data, method = 'ward'))
plt.show()

from sklearn.cluster import AgglomerativeClustering 
cluster = AgglomerativeClustering(n_clusters=4, metric= 'euclidean', linkage = 'ward') 
cluster.fit_predict(patient_data) 
plt.figure(figsize=(10, 7))
plt.scatter(patient_data[:, 0], patient_data[:, 1], c = cluster.labels_, cmap='rainbow')
plt.show()