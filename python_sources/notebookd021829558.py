#!/usr/bin/env python
# coding: utf-8

# From
# 
# https://corpocrat.com/2014/08/29/tutorial-titanic-dataset-machine-learning-for-kaggle/

# In[ ]:


import numpy as np
import pandas as pd
import re as re
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn import cross_validation
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/train.csv',header=0)


# In[ ]:


df.info()


# In[ ]:


cols = ['Name','Ticket','Cabin']
df = df.drop(cols,axis=1)


# In[ ]:


df.info()


# In[ ]:


df = df.dropna()


# In[ ]:


dummies = []
cols = ['Pclass','Sex','Embarked']
for col in cols:
 dummies.append(pd.get_dummies(df[col]))


# In[ ]:


titanic_dummies = pd.concat(dummies, axis=1)


# In[ ]:


df = pd.concat((df,titanic_dummies),axis=1)


# In[ ]:


df = df.drop(['Pclass','Sex','Embarked'],axis=1)


# In[ ]:


df['Age'] = df['Age'].interpolate()


# In[ ]:


X = df.values
y = df['Survived'].values


# In[ ]:


X = np.delete(X,1,axis=1)


# In[ ]:


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.3,random_state=0)


# In[ ]:


from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=5)
clf.fit(X_train,y_train)
clf.score(X_test,y_test)


# In[ ]:


clf.feature_importances_


# In[ ]:


from sklearn import ensemble
clf = ensemble.RandomForestClassifier(n_estimators=100)
clf.fit (X_train, y_train)
clf.score (X_test, y_test)


# In[ ]:


clf = ensemble.GradientBoostingClassifier()
clf.fit (X_train, y_train)
clf.score (X_test, y_test)


# In[ ]:


#clf = ensemble.GradientBoostingClassifier(n_estimators=50)
#clf.fit(X_train,y_train)
#clf.score(X_test,y_test)


# In[ ]:


def get_Kagle(name):
 df = pd.read_csv(name,header=0)
 cols = ['Name','Ticket','Cabin']
 df = df.drop(cols,axis=1)  
 dummies = []
 cols = ['Pclass','Sex','Embarked']
 for col in cols:
  dummies.append(pd.get_dummies(df[col]))
 titanic_dummies = pd.concat(dummies, axis=1)
 df = pd.concat((df,titanic_dummies),axis=1) 
 df = df.drop(['Pclass','Sex','Embarked'],axis=1)
 df['Age'] = df['Age'].interpolate()
 X = df.values
 X = np.delete(X,1,axis=1)
 return X
 #y = df['Survived'].values
 


# In[ ]:


X_results=get_Kagle('../input/train.csv')
y_results = clf.predict(X_results)
output = np.column_stack((X_results[:,0],y_results))
df_results = pd.DataFrame(output.astype('int'),columns=['PassengerID','Survived'])
df_results.to_csv('titanic_results.csv',index=False)


# In[ ]:


df_results


# In[ ]:




