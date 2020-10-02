# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.pandas.read_csv("../input/data.csv")
# -*- coding: utf-8 -*-
f1_scores={}
X=data.iloc[:,2:32].values
y=data.iloc[:,1].values
from sklearn.preprocessing import LabelEncoder
label_encoder_y=LabelEncoder()
y=label_encoder_y.fit_transform(y)
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)
def F1score_calc(cm,s):
    precision=(cm[0][0])/(cm[0][0]+cm[1][0])
    recall=(cm[0][0])/(cm[0][0]+cm[0][1])
    f1_score=(2*recall*precision)/(recall+precision)
    f1_scores[s]=f1_score
    return f1_score
#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
cm_LR=confusion_matrix(y_test,y_pred)
F1score_calc(cm_LR,'Logistic Regression')
#SVM with linear kernel
from sklearn.svm import SVC
classifier=SVC(kernel='linear',random_state=0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
cm_SVM1=confusion_matrix(y_test,y_pred)
F1score_calc(cm_SVM1,'SVM(Linear)')
#SVM with non linear kernel
from sklearn.svm import SVC
classifier=SVC(kernel='rbf',random_state=0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
cm_SVM2=confusion_matrix(y_test,y_pred)
F1score_calc(cm_SVM2,'SVM(RBF)')
#DEcision Trees
from sklearn.tree import DecisionTreeClassifier
dtclassifier=DecisionTreeClassifier(criterion='entropy',random_state=2)
dtclassifier.fit(X_train,y_train)
y_pred=dtclassifier.predict(X_test)
cm_DT=confusion_matrix(y_test,y_pred)
F1score_calc(cm_DT,'Decision Trees')
#Random Forests
from sklearn.ensemble import RandomForestClassifier
rfclassifier=RandomForestClassifier(n_estimators=30,criterion='entropy',random_state=0)
rfclassifier.fit(X_train,y_train)
y_pred=rfclassifier.predict(X_test)
cm_RF=confusion_matrix(y_test,y_pred)
F1score_calc(cm_RF,'Random Forests')
#K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
y_pred=rfclassifier.predict(X_test)
cm_KNN=confusion_matrix(y_test,y_pred)
F1score_calc(cm_KNN,'K Nearest Neighbors')
#Naive Bayes
from sklearn.naive_bayes import GaussianNB
nbclassifier=GaussianNB()
nbclassifier.fit(X_train,y_train)
y_pred=rfclassifier.predict(X_test)
cm_NB=confusion_matrix(y_test,y_pred)
F1score_calc(cm_NB,'Naive Bayes')
#Plotting the F1scores of each of the algorithms
plt.rc('font',size=8)
plt.rc('axes',titlesize=8)
plt.rc('axes',labelsize=8)
plt.xlabel('Algorithms')
plt.xticks(rotation=90)
plt.ylabel('F1 Scores')
plt.title('Comparison of Algorithms based on F1 scores')
plt.scatter(f1_scores.keys(),f1_scores.values())






