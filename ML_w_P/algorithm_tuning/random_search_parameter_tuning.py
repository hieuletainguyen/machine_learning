import numpy as np 
from pandas import read_csv 
from scipy.stats import uniform 
from sklearn.linear_model import Ridge 
from sklearn.model_selection import RandomizedSearchCV  

path = "/workspaces/Agiat_Ikazinat/machine_learning/ML_w_P/pima-indians-diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names = headernames) 
array = data.values 
x = array[:, 0:8] 
y = array[:, 8]

param_grid = {"alpha": uniform()}
model = Ridge() 
random_search = RandomizedSearchCV(estimator= model, param_distributions = param_grid, n_iter = 50)
random_search.fit(x, y)

print(random_search.best_score_)
print(random_search.best_estimator_.alpha)