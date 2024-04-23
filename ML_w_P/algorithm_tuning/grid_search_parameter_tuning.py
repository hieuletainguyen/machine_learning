import numpy as np 
from pandas import read_csv 
from sklearn.linear_model import Ridge 
from sklearn.model_selection import GridSearchCV 

path = "/workspaces/Agiat_Ikazinat/machine_learning/ML_w_P/pima-indians-diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names = headernames) 
array = data.values 
x = array[:, 0:8] 
y = array[:, 8]

alphas = np.array([1, 0.1, 0.01, 0.001, 0.0001, 0])
param_grid = dict(alpha = alphas) 
model = Ridge()
grid = GridSearchCV(estimator = model, param_grid = param_grid) 
grid.fit(x, y)
print(grid.best_score_) 
print(grid.best_estimator_.alpha)