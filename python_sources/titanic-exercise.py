#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d
import seaborn as sns
from sklearn import svm
from sklearn.metrics import r2_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


titan = pd.read_csv('/kaggle/input/titanic/train.csv')
titan.head()


# PClass -  class 1, 2 or 3. 1 being most expensive
# SibSP - number of siblings and spouces on board.
# Parch - # parents/children onboard.

# In[ ]:


titan = titan.drop(['Name'], axis=1)
titan = titan.drop(['PassengerId'], axis=1)
titan.dtypes


# In[ ]:


titan.isnull().sum()


# * Age and Cabin number seems ot have a lot of null values.
# * Embarked has 2 null values. Need to probably replace with the most common point of embarkation
# * Age needs to be replaced by mean.
# * Shoud we drop Cabin? or replace with most common? Which class does most Nan cabin occurs?

# In[ ]:


titan.describe().transpose()


# In[ ]:


#print(titan[titan.Age != 'Nan'].Age.mean())
#print(titan[pd.notna(titan.Age)].Age.mean())
print(titan.Age.mean())

titan[pd.isna(titan.Age)].Age = titan.Age.mean()


titan.isnull().sum()
#pd.notna(titan.Age).Age.mean() = data[data.Experience >= 0].Experience.mean()
#data.Experience.mean()


# In[ ]:


rep_0 = SimpleImputer() ##using default values. missing_values = "NaN", method=mean
titan.Age = pd.DataFrame(rep_0.fit_transform(titan[['Age']]).ravel()) ## Simple imputing Age. ravel() is used when you have a 1D dataframe.


#titan.columns = cols
#titan[titan.Survived == 1].count()
#titan.isnull().sum()


# In[ ]:


sns.pairplot(titan, hue='Survived', diag_kind='hist')


# In[ ]:


sns.heatmap(titan.corr(),annot=True,cmap='YlGnBu')


# Survival seems most closely correlated to Class and Fare.
# Lower class have lower survival rates.

# In[ ]:


sns.lmplot(x='Fare', y='Pclass', data=titan, hue='Survived')


# In[ ]:


sns.jointplot(titan.Pclass, titan.Fare, kind='kde')


# In[ ]:


X=titan[['Pclass','Fare']]
X_train,X_test, y_train, y_test = train_test_split(titan[['Pclass','Fare']], titan.Survived, test_size=0.3, stratify=titan.Survived, random_state=7)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# In[ ]:


LGR = LogisticRegression(solver='liblinear')
LGR.fit(X_train,y_train)
LGR_predict = LGR.predict(X_test)
print("Logistic Regression Score is", LGR.score(X_test,y_test))
sns.heatmap(metrics.confusion_matrix(y_test, LGR_predict), annot=True)


# In[ ]:


knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_y = knn.predict(X_test)
print("KNN Score predictor is ", knn.score(X_test,y_test))
sns.heatmap(metrics.confusion_matrix(y_test, knn_y), annot=True)
###Feature scaling does improve performance of KNN. But not the Logistic regression. Wonder why? And the model performs better than Logistic! 
### Is it because of the type classification? Maybe only because it is 2 class. Or is it quality of data.


# In[ ]:


svc = svm.SVC(kernel='rbf',C=10,gamma='auto')
svc.fit(X_train, y_train)
svc_y = svc.predict(X_test)
print("SVM Score predictor is ", svc.score(X_test,y_test))
sns.heatmap(metrics.confusion_matrix(y_test, svc_y), annot=True)

