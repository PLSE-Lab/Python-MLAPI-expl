#!/usr/bin/env python
# coding: utf-8

# Importing libraries and input datasets.

# In[ ]:


# Importing packages
import numpy as np
import pandas as pd
import os
from pandas import DataFrame,Series
from sklearn import tree
import matplotlib
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn import neighbors
from sklearn import linear_model
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
test2 = pd.read_csv("../input/test.csv")


# In[ ]:


train.describe()


# In[ ]:


train.head()


# In[ ]:


sex = np.zeros(len(train))
sex[train['Sex']== 'male'] = 1
sex[train['Sex']== 'female'] = 0
train['Sex'] = sex

train.head()


# In[ ]:


train['Embarked'].fillna(train['Embarked'].mode()[0],inplace=True)
Embarked = np.zeros(len(train))
Embarked[train['Embarked']== 'C'] = 1
Embarked[train['Embarked']== 'Q'] = 2
Embarked[train['Embarked']== 'S'] = 3
train['Embarked'] = Embarked
train.head()


# In[ ]:


print(type(train))
dropping = ['PassengerId','Name', 'Ticket','Cabin']
train.drop(dropping,axis=1, inplace=True)
test.drop(dropping,axis=1, inplace=True)
train.head()


# In[ ]:


train['Age'].fillna(train['Age'].mean(),inplace=True)
age = np.zeros(len(train))
age[train['Age']<20] = 1
age[(train['Age']>=20)&(train['Age']<60)] = 2
age[(train['Age']>=60)] = 3
train['Age'] = age
train.head()


# In[ ]:


from sklearn.model_selection import cross_val_score, KFold

train_y=train['Survived']
train_ft=train.drop('Survived',axis=1)
kf = KFold(n_splits=10,random_state=1)
print(train_ft.head())
print(train_y.head())


# In[ ]:


from sklearn.svm import SVC, LinearSVC
svc = SVC(C = 30, gamma = 0.01)
svc.fit(train_ft, train_y) 

acc_SVM = cross_val_score(svc,train_ft,train_y,cv=kf)
print(acc_SVM.mean())


# In[ ]:


from sklearn.neighbors.nearest_centroid import NearestCentroid
NN = NearestCentroid()
NN.fit(train_ft, train_y)
acc_NN = cross_val_score(NN,train_ft,train_y,cv=kf)
print(acc_NN.mean())


# In[ ]:


sex_test = np.zeros(len(test))
sex_test[test['Sex']== 'male'] = 1
sex_test[test['Sex']== 'female'] = 0
test['Sex'] = sex_test


# In[ ]:


test['Embarked'].fillna(test['Embarked'].mode()[0],inplace=True)
Embarked = np.zeros(len(test))
Embarked[test['Embarked']== 'C'] = 1
Embarked[test['Embarked']== 'Q'] = 2
Embarked[test['Embarked']== 'S'] = 3
test['Embarked'] = Embarked


# In[ ]:


age = np.zeros(len(test))
age[test['Age']<20] = 1
age[(test['Age']>=20)&(test['Age']<60)] = 2
age[(test['Age']>=60)] = 3
test['Age'] = age
test['Fare'].fillna(test['Fare'].mean(),inplace=True)
test.head()


# In[ ]:


np.sum(test.isnull())


# In[ ]:


predictions = svc.predict(test)
print(predictions)


# In[ ]:


submission = pd.DataFrame({ 'PassengerId': test2['PassengerId'],
                            'Survived': predictions })
submission.to_csv("submission.csv", index=False)

