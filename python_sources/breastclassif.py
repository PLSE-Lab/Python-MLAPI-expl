# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import svm # SVM classifier
from sklearn.neural_network import MLPClassifier # MLP classifier
from sklearn import cross_validation # used to test classifier
from sklearn.cross_validation import KFold, cross_val_score, train_test_split
from sklearn import metrics

import matplotlib.pyplot as plt
from pandas.tools.plotting import andrews_curves #visu


#input
df = pd.read_csv("../input/data.csv", header = 0)
#data formating
df = df.drop("id", 1)
df = df.drop("Unnamed: 32", 1)

plt.figure()
andrews_curves(df, "diagnosis")

# M: classe 1 B: classe 0
df.diagnosis.unique()
d = {'M' : 1, 'B' : 0}
df['diagnosis'] = df['diagnosis'].map(d)

#setting features
features = list(df.columns[1:31])
#setting data
X = df[features]
#setting target
y = df["diagnosis"]

#setting svm classifier
svc = svm.SVC(kernel='linear', C=1).fit(X, y)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100), random_state=1).fit(X, y)

#dividing data to have a training and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .4, random_state=0)

#SVM cross validation
Kfold = KFold(len(df), n_folds=10, shuffle=False)
print("KfoldCrossVal mean score using SVM is %s" %cross_val_score(svc,X,y,cv=10).mean())
#SVM metrics
sm = svc.fit(X_train, y_train)
y_pred = sm.predict(X_test)
print("Accuracy score using SVM is %s" %metrics.accuracy_score(y_test, y_pred))

#MLP cross validation
Kfold = KFold(len(df), n_folds=10, shuffle=False)
print("KfoldCrossVal mean score using MLP is %s" %cross_val_score(clf,X,y,cv=10).mean())
#MLP metrics
cm = clf.fit(X_train,y_train)
y_pred = cm.predict(X_test)
print("Accuracy score using MLP is %s" %metrics.accuracy_score(y_test, y_pred))




