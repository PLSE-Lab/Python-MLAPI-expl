#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# First we will import Pandas lib 

# In[ ]:


import pandas as pd


# reading the head of the data set.

# In[ ]:


df=pd.read_csv('../input/glass/glass.csv')


# In[ ]:


df.head()


# now.we can import some visulization library like matplotlib and seaborn and also we will import numpy.
# 

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# first we have to understand the data.

# In[ ]:


df.info()


# NO, null values are there,that is a good sign.
# 

# In[ ]:


df.shape


# In[ ]:


sns.countplot(df['Type'])


# we have to take care here about one thing that is ,
# training set is small.
# so what we will do ,that we will train and clsiify roughly first and then we can improve over it considering over or under fitting.

# In[ ]:


from sklearn.preprocessing import StandardScaler


# to normalize the data set

# In[ ]:


X=df.drop('Type',axis=1)
y=df['Type']


# In[ ]:


scaler=StandardScaler()


# 

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.35,random_state=101)


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


model=DecisionTreeClassifier()


# In[ ]:


model.fit(X_train,y_train)


# In[ ]:


prd=model.predict(X_test)


# In[ ]:


prd_train=model.predict(X_train)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(prd,y_test))


# In[ ]:


print(classification_report(prd_train,y_train))


# we are not achieving much high accuracy with these models ,we can try for DNN next.

# In[ ]:




