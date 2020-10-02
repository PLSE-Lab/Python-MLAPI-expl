#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[ ]:


data = "../input/homeprices1.csv"


# In[ ]:


df = pd.read_csv(data)


# In[ ]:


df


# In[ ]:


df.bedrooms.mean()


# In[ ]:


br = math.floor(df.bedrooms.mean())


# In[ ]:


br


# In[ ]:


df.bedrooms = df.bedrooms.fillna(br)


# In[ ]:


df


# In[ ]:


reg = linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']], df.price)


# In[ ]:


reg.coef_


# In[ ]:


reg.intercept_


# In[ ]:


reg.predict([[3000,3,40]])


# In[ ]:


reg.predict([[3200,4,18]])

