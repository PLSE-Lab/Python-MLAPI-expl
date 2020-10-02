#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


data = pd.read_csv('/kaggle/input/fish-market/Fish.csv')


# Today I'm trying to predict the type of species using Logistic Regression . IT worked out pretty well
# 
# 
# First let use see if the dataset requires any cleaning

# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.isnull().sum()


# There are no null values in the data set . So now we can get into prediction process

# Let us see how many species are there in the dataset

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 6))
sns.set(style="darkgrid")
ax = sns.countplot(x="Species", data=data)


# In[ ]:


x= sns.PairGrid(data)
x = x.map(plt.scatter)


# In[ ]:


X = data['Species']
y=data['Height']
z=data['Width']

plt.figure(figsize=(16, 6))

plt.scatter(X,y,label='Height')
plt.scatter(X,z,label = 'Width')
plt.legend()
plt.show()
data.Species.unique()


# I will be using Logistic Regresion for this problem

# In[ ]:


from sklearn.linear_model import *
from sklearn.model_selection import train_test_split

X = data.drop(columns=['Species'])
y = data.Species


# Split the data by test size of 20 percent . I tried different sizes of the test and train size . It seems like 20 percent is the best fit

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()


# Train the model 

# In[ ]:


model.fit(X_train,y_train)


# Now lets predict the value

# In[ ]:


model.predict(X_test)


# In[ ]:


model.score(X_test,y_test)


# We got 87 percent accuracy which is perfectly amazing . 
# 
# The reason i used LOgistics regression is it works well with labels and small taining sets

# In[ ]:




