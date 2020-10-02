#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math
import pandas as pd
from scipy import stats


# In[2]:


pth="../input/data-train-reducedcsv/Data_train_reduced.csv"


# In[3]:


data=pd.read_csv(pth)


# In[4]:


drop_colums = [
    'Product',
    'q8.7',
    'q8.9',
    'q8.10',
    'q8.17',
    'q8.18',
    'Respondent.ID',
    's7.involved.in.the.selection.of.the.cosmetic.products',
    'q8.2',
    'q8.7',
    'q8.8',
    'q8.9',
    'q8.10',
    'q8.12',
    'q8.17',
    'q8.18',
    'q8.20',
    's13.2',
    'q8.13',
    'q7',
    'q8.1',
    'q8.5',
    'q8.6',
    'q8.11',
    'q8.13',
    'q8.19'
]


# In[5]:


data=data.drop(drop_colums,axis=1)


# In[6]:


print(data.columns)


# In[7]:


X=data['Instant_Liking']


# In[8]:


c = data.corr()


# In[9]:


s = c.unstack()


# In[10]:


so = s.sort_values(kind="quicksort")


# In[11]:


print(so)


# In[12]:


Y=data['q1_1.personal.opinion.of.this.Deodorant']


# In[13]:


coef=np.corrcoef(X, Y)
print(coef)
plt.scatter(X, Y)
plt.show()


# In[14]:


mean_x = np.mean(X)
mean_y = np.mean(X)


# In[15]:


lineeq=stats.linregress(X,Y)


# In[16]:


print(lineeq)
b0=lineeq[0]
b1=lineeq[1]


# In[36]:


max_x = np.max(X)
min_x = np.min(X)
x = np.linspace(max_x, min_x, 2500)
y = b0 + b1 * x


# In[37]:


plt.plot(x, y, color='#58b970', label='Regression Line')
plt.scatter(X, Y, c='#ef5423', label='Scatter Plot')
plt.xlabel('q1_1.personal.opinion.of.this.Deodorant')
plt.ylabel('Instant_Liking')
plt.legend()
plt.show()


# In[39]:


rmse = 0
for i in range(len(X)):
    y_pred = b0 + b1 * X[i]
    rmse += (Y[i] - y_pred) ** 2
rmse = np.sqrt(rmse/len(X))
print(rmse)


# In[40]:


ss_t = 0
ss_r = 0
for i in range(len(X)):
    y_pred = b0 + b1 * X[i]
    ss_t += (Y[i] - mean_y) ** 2
    ss_r += (Y[i] - y_pred) ** 2
r2 = 1 - (ss_r/ss_t)
print(r2)

