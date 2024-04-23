from pandas import read_csv 
from sklearn.model_selection import KFold, cross_val_score 
from sklearn.ensemble import ExtraTreesClassifier

path = "/workspaces/Agiat_Ikazinat/machine_learning/ML_w_P/pima-indians-diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names = headernames) 
array = data.values 
x = array[:, 0:8] 
y = array[:, 8]

kfold = KFold(n_splits = 10) 

num_trees = 150 
max_features = 5 
model = ExtraTreesClassifier(n_estimators = num_trees, max_features = max_features) 
results = cross_val_score(model, x, y, cv=kfold)
print(results.mean())
