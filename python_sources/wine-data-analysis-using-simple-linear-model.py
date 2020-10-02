#!/usr/bin/env python
# coding: utf-8

# In[50]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model


# In[51]:


pth="../input/linear.csv"


# In[52]:


data=pd.read_csv(pth)


# In[53]:


X=data['x']
Y=data['y']


# In[54]:


c = data.corr()
s = c.unstack()
so = s.sort_values(kind="quicksort")
print(so)


# In[78]:


coef=np.corrcoef(X, Y)
print(coef)
plt.scatter(X, Y)
plt.show()


# In[89]:


X = X.values.reshape(-1,1)
reg=linear_model.LinearRegression()
reg.fit(X,Y)


# In[90]:


m=reg.coef_[0]
b=reg.intercept_
print("slope=",m, "intercept=",b)


# In[94]:


plt.scatter(X,Y,color='black')
predicted_values = [reg.coef_ * i + reg.intercept_ for i in Y]
plt.plot(X, predicted_values, 'purple')
plt.xlabel("height")
plt.ylabel("weight")


# In[ ]:




