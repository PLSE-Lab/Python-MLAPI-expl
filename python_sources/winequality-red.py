#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[ ]:


import pandas as pd


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


data = pd.read_csv('../input/winequality-red.csv')


# In[ ]:


data.head(5)


# In[ ]:


data.groupby('quality').count()


# In[ ]:


data.isna().sum()


# In[ ]:


plt.figure(figsize=(8,7))
plt.scatter(data['pH'],data['quality'])
plt.xlabel('PH level')
plt.ylabel('Quality')


# In[ ]:


Y=data['quality']
mapping={3:'a',4:'b',5:'c',6:'d',7:'e',8:'f'}
data
X=data.drop(columns=['quality'])


# In[ ]:


clf = LinearRegression()


# In[ ]:


xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.3)


# In[ ]:


clf.fit(xtrain,ytrain)


# In[ ]:


clf.predict(xtest)


# In[ ]:


clf.score(xtest,ytest)


# In[ ]:




