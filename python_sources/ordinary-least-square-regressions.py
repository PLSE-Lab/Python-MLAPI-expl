#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import missingno
import os
sns.set()


# In[ ]:


df = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")
df.head()


# In[ ]:


df.shape


# In[ ]:


missingno.matrix(df)


# In[ ]:


df.info()


# In[ ]:


data = df[['price', 'bedrooms', 'bathrooms','sqft_living', 'sqft_lot', 'floors', 'condition', 'grade','yr_built']]


# In[ ]:


data = data[:100]


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


y = data['price']
x1 = data[['sqft_living', 'condition', 'grade','yr_built']]


# In[ ]:


plt.scatter(data['sqft_living'], y)
plt.xlabel('size', fontsize=20)
plt.ylabel('price', fontsize=20)
plt.show()


# In[ ]:


x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
results.summary()


# In[ ]:


import pandas as pd
kc_house_data = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")

