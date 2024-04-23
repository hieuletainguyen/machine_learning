from pandas import read_csv 
from sklearn.model_selection import KFold, cross_val_score 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import SVC 
from sklearn.ensemble import VotingClassifier 

path = "/workspaces/Agiat_Ikazinat/machine_learning/ML_w_P/pima-indians-diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names = headernames) 
array = data.values 
x = array[:, 0:8] 
y = array[:, 8]

kfold = KFold(n_splits = 10)

estimators = []
model1 = LogisticRegression() 
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier() 
estimators.append(('cart', model2))
model3 = SVC() 
estimators.append(('svm', model3))

# create the voting exnsemble model by cimbingn the predictions of above created sub models
ensemble = VotingClassifier(estimators) 
results = cross_val_score(ensemble, x, y, cv=kfold) 
print(results.mean())