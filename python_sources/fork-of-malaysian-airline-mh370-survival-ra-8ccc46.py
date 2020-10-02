#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import math
import seaborn as sns

#from sklearn.model_selection import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
flight_data=pd.read_csv('../input/Flight.csv')
flight_data=flight_data.fillna(0)
print(flight_data.head())


# In[ ]:


#There are 239 rows and 13 cloumns
flight_data.shape
flight_data.describe


# In[ ]:


flight_data.info()


# In[ ]:


#correlation
sns.heatmap(flight_data.corr())


# In[ ]:


sns.countplot(x="survived",data=flight_data)


# In[ ]:


sns.countplot(x="survived",hue="Sex",data=flight_data)


# In[ ]:


sns.countplot(x="survived",hue="Class",data=flight_data)


# In[ ]:


sns.countplot(x="Age",hue="Class",data=flight_data)


# In[ ]:


flight_data["Age"].plot.hist()


# In[ ]:


flight_data["Class"].plot.hist()


# In[ ]:


flight_data["Fare"].plot.hist()


# In[ ]:


flight_data.info()


# In[ ]:


flight_data.isnull().sum()


# In[ ]:


sns.boxplot(x="Class",y="Age",data=flight_data)


# In[ ]:


sns.boxplot(x="Class",y="Sex",data=flight_data)


# In[ ]:


flight_data.head()


# In[ ]:


sex=pd.get_dummies(flight_data["Sex"],drop_first=True)
sex.head(5)


# In[ ]:


pclass=pd.get_dummies(flight_data["Class"],drop_first=True)
pclass.head(5)


# In[ ]:


flight_data.drop(['First name','Last name',"Sex","Class","Nationality",'Ticket','Fare','cabin',"Embarked","body"],axis=1,inplace=True)
flight_data=pd.concat([flight_data,pclass,sex],axis=1)
flight_data.head()


# In[ ]:


X=flight_data.drop("survived",axis=1)
y=flight_data["survived"]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)


# In[ ]:


logmodel=LogisticRegression(solver='liblinear', n_jobs=1)
logmodel.fit(X_train,y_train)
y_predict=logmodel.predict(X_test)


# In[ ]:


accuracy_score=accuracy_score(y_test,y_predict)
print (accuracy_score*100)


# In[ ]:




