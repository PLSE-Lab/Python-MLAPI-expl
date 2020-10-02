#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


# import libraries:
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt


# In[ ]:


#import dataset:
df=pd.read_csv("../input/Admission_Predict.csv")


# In[ ]:


df.head()


# In[ ]:


# visualization of dataset:
sns.pairplot(df,hue='Research')


# In[ ]:


df.shape


# In[ ]:


sns.scatterplot(df['University Rating'],df['Chance of Admit '],hue='Research',data=df)


# In[ ]:


sns.distplot(df['GRE Score'],bins=20,kde=False)


# In[ ]:


sns.distplot(df['TOEFL Score'],bins=20,kde=False)


# In[ ]:


sns.lmplot('GRE Score','TOEFL Score',hue='Research',data=df)


# In[ ]:


sns.heatmap(df)


# In[ ]:


sns.lineplot('GRE Score','TOEFL Score',hue='Research',data=df)


# In[ ]:


sns.scatterplot(df['GRE Score'],df['CGPA'],hue='Research',data=df)


# In[ ]:


sns.scatterplot(df['TOEFL Score'],df['CGPA'],hue='Research',data=df)


# In[ ]:


df=df.drop('Serial No.',axis=1)


# In[ ]:


df.head()


# In[ ]:


# make the dataset into the dependent and independent:
X=df.iloc[:,:-1].values
y=df.iloc[:,7].values


# In[ ]:


X


# In[ ]:


y


# In[ ]:


# spliting the datatset into the train and test set:
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[ ]:


# feacture sacling :
#rom sklearn.preprocessing import StandardScaler
#c=StandardScaler()
#_train=sc.fit_transform(X_train)
#_test=sc.fit_transform(X_test)


# In[ ]:


# fitting the dataset into the model Logistic_regression:
from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(X_train,y_train)


# In[ ]:


from sklearn.model_selection import cross_val_score
result_score=cross_val_score(regression,X_train,y_train,cv=5).mean()


# In[ ]:


result_score


# In[ ]:


score=regression.score(X_test,y_test)


# In[ ]:


score

