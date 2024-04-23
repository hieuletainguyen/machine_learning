import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

dataset = pd.read_csv(path, names = headernames) 
print(dataset.head())

x = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, 4].values 

from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4) 

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() 
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test) 

from sklearn.neighbors import KNeighborsClassifier 
classifier = KNeighborsClassifier(n_neighbors = 8) 
classifier.fit(x_train, y_train) 

y_pred = classifier.predict(x_test) 

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
result = confusion_matrix(y_test, y_pred) 
print("Confusion matrix: ", result)
result1 = classification_report(y_test, y_pred) 
print("classification report: ", result1) 
result2 = accuracy_score(y_test, y_pred) 
print("Accuracy: ", result2)

