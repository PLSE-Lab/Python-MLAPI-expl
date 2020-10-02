# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

#Import the libraries
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import matplotlib.ticker as ticker
from sklearn import preprocessing
import pylab as pl
import scipy.optimize as opt
#Load the data
df=pd.read_csv('C:/Users/chahd/Desktop/train.csv')
df.head()
df.columns
test_df=pd.read_csv('C:/Users/chahd/Desktop/test.csv')
test_df.columns
#drop NAN
reduced_df=df.dropna()
#Convert the Pandas to Numpy array
X=reduced_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].values
X[0:5]
Y=reduced_df['Survived'].values
Y[0:5]
#Normalization
Sex=preprocessing.LabelEncoder()
Sex.fit(['male','female'])
X[:,1]=Sex.transform(X[:,1])
X[0:5]
reduced_df['Embarked'].value_counts()
Embarked=preprocessing.LabelEncoder()
Embarked.fit(['S','C','Q'])
X[:,6]=Embarked.transform(X[:,6])
X[0:5]
#Classification
 #KNN
from sklearn.neighbors import KNeighborsClassifier
K=4
neigh=KNeighborsClassifier(n_neighbors=K).fit(X,Y)
neigh
rtest=test_df.dropna()
test=rtest[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].values
Sex=preprocessing.LabelEncoder()
Sex.fit(['male','female'])
test[:,1]=Sex.transform(test[:,1])
Embarked=preprocessing.LabelEncoder()
Embarked.fit(['S','C','Q'])
test[:,6]=Embarked.transform(test[:,6])
Y_pred=neigh.predict(test)
Y_pred[0:5]
from sklearn import metrics
print("Train set accuracy",metrics.accuracy_score(Y,neigh.predict(X)))
 #Decision Trees
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=3)
X_train.shape        
X_test.shape
Y_train.shape
Y_test.shape
TitanicTree=DecisionTreeClassifier(criterion="gini",max_depth=4)
TitanicTree
TitanicTree.fit(X_train,Y_train)
predTree=TitanicTree.predict(X_test)
predTree[0:5]
Y_test[0:5]
metrics.accuracy_score(Y_test,predTree)
  #Log Regression 
from sklearn.linear_model import LogisticRegression

LR=LogisticRegression(C=0.01,solver='liblinear').fit(X_train,Y_train)
yhat=LR.predict(X_test)
yhat
yhat_prob=LR.predict_proba(X_test)
yhat_prob
from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(Y_test,yhat)
  # SVM
from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, Y_train) 
yhat = clf.predict(X_test)
yhat
yhat [0:5]
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(Y_test, yhat)
f1_score(Y_test, yhat, average='weighted')
