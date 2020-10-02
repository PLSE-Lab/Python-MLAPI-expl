#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df=pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")


# In[ ]:


df


# In[ ]:


df.head()


# In[ ]:


df.isna().sum()


# In[ ]:


df.describe()


# In[ ]:


sns.barplot(df['quality'],df['fixed acidity'])


# In[ ]:


sns.barplot(df['quality'],df['chlorides'])


# In[ ]:


bins = (2, 6.5, 8)
group_names = ['bad', 'good']
df['quality'] = pd.cut(df['quality'], bins = bins, labels = group_names)


# In[ ]:


df['quality'].unique()


# In[ ]:


df['quality']


# In[ ]:


df.info()


# In[ ]:


from sklearn.preprocessing import StandardScaler, LabelEncoder
encoder=LabelEncoder()
df['quality']=encoder.fit_transform(df['quality'])
#df=pd.get_dummies(df,drop_first=True)


# In[ ]:


df['quality'].value_counts()


# In[ ]:


sns.countplot(df['quality'])


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df.drop('quality',axis=1),df['quality'])


# In[ ]:


type(y_train)


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train1=scaler.fit_transform(x_train)
x_test1=scaler.transform(x_test)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model=LogisticRegression()
model.fit(x_train1,y_train)
z=model.predict(x_test1)
scores=cross_val_score(model,df.drop("quality",axis=1),df['quality'],cv=7).mean()
print("Cross Validation score={}".format(scores))
print(classification_report(y_test,z))
print(confusion_matrix(y_test,z))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(x_train1,y_train)
z=model.predict(x_test1)
scores=cross_val_score(model,df.drop("quality",axis=1),df['quality'],cv=7).mean()
print("Cross Validation score={}".format(scores))
print(classification_report(y_test,z))
print(confusion_matrix(y_test,z))


# RANDOM FOREST WITHOUT SCALING

# In[ ]:


model=RandomForestClassifier()
model.fit(x_train,y_train)
z=model.predict(x_test)
scores=cross_val_score(model,df.drop("quality",axis=1),df['quality'],cv=7).mean()
print("Cross Validation score={}".format(scores))
print(classification_report(y_test,z))
print(confusion_matrix(y_test,z))


# In[ ]:


from sklearn.svm import SVC
model=SVC()
model.fit(x_train1,y_train)
z=model.predict(x_test1)
scores=cross_val_score(model,df.drop("quality",axis=1),df['quality'],cv=7).mean()
print("Cross Validation score={}".format(scores))
print(classification_report(y_test,z))
print(confusion_matrix(y_test,z))


# In[ ]:


get_ipython().run_line_magic('pinfo', 'SVC')


# In[ ]:


param={'C':[0.01,0.05,0.1,0.5,1],
       'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],
      'kernel':['linear', 'rbf']}
from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(model,param_grid=param,scoring='accuracy',cv=10)
grid.fit(x_train1,y_train)
grid.best_params_


# In[ ]:


model=SVC(C=1, gamma=1, kernel='rbf')
model.fit(x_train,y_train)
z=model.predict(x_test)
scores=cross_val_score(model,df.drop("quality",axis=1),df['quality'],cv=7).mean()
print("Cross Validation score={}".format(scores))
print(classification_report(y_test,z))
print(confusion_matrix(y_test,z))


# In[ ]:




