import numpy as np 
import pandas as pd 
path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
data = pd.read_csv(path, names = headernames) 
array = data.values 
x = array[:, :2] 
y = array[:, 2] 
print(data.shape) 

from sklearn.neighbors import KNeighborsRegressor 
knnr = KNeighborsRegressor(n_neighbors = 10) 
knnr.fit(x, y) 

print("The MSE is: ", np.power(y - knnr.predict(x), 2).mean()) 
