#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np 
from sklearn import linear_model
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv(r'../input/kc_house_data.csv')
df.head()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.sqft_lot,df.price,color='blue',marker='+')


# In[ ]:


x = df[['bedrooms','bathrooms','sqft_lot','floors']]
x.head()


# In[ ]:


y = df.price
y.head()


# In[ ]:


# Create linear regression object
reg = linear_model.LinearRegression()
reg.fit(x,y)


# In[ ]:


reg.predict([[3,1.00,5650,1.0]])


# In[ ]:


reg.coef_


# In[ ]:


reg.intercept_


# In[ ]:


p = reg.predict(x)
p


# In[ ]:


final = df[['bedrooms','bathrooms','sqft_lot','floors']]
final.head()


# In[ ]:


final['price'] = p
final.head()


# In[ ]:


'''We have completed .... In a Simple Cleaner Code'''


# In[ ]:




