import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score


import matplotlib.pyplot as plt



iris = datasets.load_iris()
X = iris.data[:,:]
y = iris.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
print('Training data')
print(X_train_std)
print('Training Ouput')
print(y_train)

ppn = Perceptron(max_iter = 40,eta0 = 0.1,random_state=1)
ppn.fit(X_train_std,y_train)
y_pred = ppn.predict(X_test_std)
print('Accuaracy error is :',accuracy_score(y_pred,y_test))

plt.scatter(y_pred,y_test)
plt.show()

    




