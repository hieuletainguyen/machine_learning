from pandas import read_csv 
from sklearn.model_selection import KFold, cross_val_score 
from sklearn.ensemble import BaggingClassifier 
from sklearn.tree import DecisionTreeClassifier 

path = "/workspaces/Agiat_Ikazinat/machine_learning/ML_w_P/pima-indians-diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names = headernames) 
array = data.values 
x = array[:, 0:8] 
y = array[:, 8]

seed = 7 
kfold = KFold(n_splits = 10) 
cart = DecisionTreeClassifier() 

num_trees = 150 
model = BaggingClassifier(estimator = cart, n_estimators = num_trees, random_state = seed) 
results = cross_val_score(model, x, y, cv=kfold) 

print(results.mean())