#!/usr/bin/env python
# coding: utf-8

# **we will do this as simple as possible leave a like or comment if it helped you**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/mushrooms.csv')


# In[ ]:


#Checking for duplicates
tot=len(set(data.index))
last=data.shape[0]-tot
last


# In[ ]:


#Checking for null values
data.isnull().sum()


# In[ ]:


#checking the shape of dataset
data.shape


# In[ ]:


#Lets see how the target variable is balanced
print(data['class'].value_counts())
sns.countplot(x='class', data=data)
plt.show()


# In[ ]:


#Looking for categorical data
cat=data.select_dtypes(include=['object']).columns
cat


# In[ ]:


#detailed view of each columns
for c in cat:
    print(c)
    print("-"*50)
    print(data[c].value_counts())
    sns.countplot(x=c, data=data)
    plt.show()
    print("-"*50)


# In[ ]:


#we will remove what all we think not important or less contribution to target
data['cap-shape']=data[data['cap-shape']!='c']
data.dropna(inplace=True)
data.shape


# In[ ]:


data['cap-surface']=data[data['cap-surface']!='g']
data.dropna(inplace=True)
data.shape


# In[ ]:


data.drop('veil-type',axis=1,inplace=True)


# In[ ]:


cat=data.select_dtypes(include='object').columns
cat


# In[ ]:


#lets convert categorical data to numerical
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in cat:
    data[i]=le.fit_transform(data[i])
    


# In[ ]:


f,ax = plt.subplots(figsize=(20, 15))
sns.heatmap(data.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)
plt.show()


# In[ ]:


#lets do some feature engineering for fun
data['f-engineer']=((data['gill-size']+5)*(data['population']+5)*(1/((data['gill-color']+5)*(data['bruises']+5)*(data['ring-type']+5))))


# In[ ]:


f,ax = plt.subplots(figsize=(20, 15))
sns.heatmap(data.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)
plt.show()


# In[ ]:


X = data.iloc[:,1:]
X = X.values
y = data['class'].values


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)


# In[ ]:


algo = {'LR': LogisticRegression(), 
        'DT':DecisionTreeClassifier(), 
        'RFC':RandomForestClassifier(n_estimators=100), 
        'SVM':SVC(gamma=0.01),
        'KNN':KNeighborsClassifier(n_neighbors=10)
       }

for k, v in algo.items():
    model = v
    model.fit(X_train, y_train)
    print('Acurracy of ' + k + ' is {0:.2f}'.format(model.score(X_test, y_test)*100)+'%')


# In[ ]:


#yes we have accuracy of 100%

