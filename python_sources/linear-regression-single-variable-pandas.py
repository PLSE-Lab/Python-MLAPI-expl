#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''Linear regression is for "prediction" else other are for 'classification'.linear regression is for for single variable as well as multiple variable both.
basic line of predction is: y = mx + c i.e price = m x area + b , where m = slope(gradient) , b = Y intercept '''


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
plt.xlabel('Square_Foot')
plt.ylabel('Price')
plt.scatter(df.sqft_living,df.price,color='red',marker='+')


# In[ ]:


area = df[['sqft_living']]
area.head()


# In[ ]:


price = df.price
price.head()


# In[ ]:


# Create linear regression object
reg = linear_model.LinearRegression()
reg.fit(area,price)


# In[ ]:


reg.predict([[3000]])


# In[ ]:


reg.coef_


# In[ ]:


reg.intercept_


# In[ ]:


'''Y = m * X + b (m is coefficient and b is intercept)'''


# In[ ]:


3000*280.6235679 + (-45380.7430944728)


# In[ ]:


df_area = df.sqft_living
df_area.head()
df_area.values.reshape(-1,1)


# In[ ]:


p = reg.predict(df_area) #Reshaped to 2D Array Even Though Not Working..Sorry.
p.head()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('area', fontsize=20)
plt.ylabel('price', fontsize=20)
plt.scatter(df.sqft_living,df.price,color='red',marker='+')
plt.plot(df.sqft_living,reg.predict(df[['sqft_living']]),color='blue')


# In[ ]:




