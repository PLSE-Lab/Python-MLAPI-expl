#!/usr/bin/env python
# coding: utf-8

# # House Prices for Beginners - Part 1 (one-feature solution)

# This notebook was prepared for the Brisbane.AI [Kaggle Hackathon](https://www.meetup.com/Brisbane-Artificial-Intelligence/events/251176784/).

# In[80]:


import os
import math

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from IPython.display import FileLink


# In[4]:


df = pd.read_csv('../input/train.csv')


# In[6]:


df.head()


# In[7]:


test_df = pd.read_csv('../input/test.csv')


# In[8]:


test_df.head()


# In[14]:


plt.scatter(x=df.GrLivArea, y=df.SalePrice)
plt.title('Sale Price vs Greater Living Area')


# In[16]:


df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF'] + df['GarageArea'] + df['EnclosedPorch'] + df['ScreenPorch']


# In[17]:


df.head()


# In[19]:


plt.scatter(x=df.TotalSF, y=df.SalePrice)
plt.title('Sale Price vs Total SF')


# The goal of linear regression is to discover an intercept and coefficient that minimises the error in this equation $\text{SalePrice} = \text{intercept} + \text{coefficient} \cdot \text{TotalSF}$
# 
# We'll calculate error using root mean squared error. All that means is for each data point (dot in the above plot), we'll calculate our equation using some learned $\text{intercept}$ and $\text{coefficient}$ values, then figure out the error by subtracting the y actual from y predicted (denoted as $\hat{y}$):
# 
# $y - \hat{y}$
# 
# Then, we'll square that result which does 2 things:
# 
# * Ensures the error is a positive value ($-1^2 = 1$)
# * Maximises larger error values
# 
# We'll then turn it into a single number by taking the mean of all our predictions, then taking the square root of that.
# 
# We can train a linear regression model using the LinearRegression class in Scikit learn.

# In[21]:


model = LinearRegression()


# In[25]:


model.fit(X=df[['TotalSF']], y=df.SalePrice)


# In[26]:


model.coef_


# In[27]:


model.intercept_


# In[29]:


predicted_values = model.intercept_ + df.TotalSF * model.coef_


# In[32]:


plt.scatter(x=df.TotalSF, y=df.SalePrice)
plt.plot(df.TotalSF, predicted_values, color='red')
plt.title('Sale Price vs Total SF (with predicted vales)')
plt.show()


# > Root Mean **Squared** **Error** (RMSE) 

# In[37]:


(-25525.212290) ** 2


# In[41]:


math.sqrt(((predicted_values - df.SalePrice) ** 2).mean())


# In[44]:


indexes_to_drop = df[(df.TotalSF > 8000) & (df.SalePrice < 400000)].index


# In[45]:


df.shape


# In[46]:


df.drop(indexes_to_drop, inplace=True)


# In[47]:


df.shape


# In[86]:


model = LinearRegression()


# In[87]:


model.fit(X=df[['TotalSF']], y=df.SalePrice)


# In[88]:


preds = model.predict(df[['TotalSF']])


# In[92]:


plt.scatter(x=df.TotalSF, y=df.SalePrice)
plt.plot(df.TotalSF, preds, color='red')
plt.title('Sale Price vs Total SF (with predicted vales - outliers removed)')
plt.show()


# In[55]:


math.sqrt(((preds - df.SalePrice) ** 2).mean())


# In[56]:


df['SalePriceLog'] = np.log(df.SalePrice)


# In[58]:


model = LinearRegression()
model.fit(X=df[['TotalSF']], y=df.SalePriceLog)


# In[59]:


preds = model.predict(df[['TotalSF']])


# In[60]:


math.sqrt(((preds - df.SalePriceLog) ** 2).mean())


# In[61]:


df.shape


# In[64]:


total_sqft_train, total_sqft_val, sale_price_train, sale_price_val = train_test_split(df[['TotalSF']], df.SalePriceLog, test_size=0.2, random_state=42)


# In[65]:


total_sqft_train.shape


# In[66]:


model = LinearRegression()
model.fit(X=total_sqft_train, y=sale_price_train)


# In[68]:


preds = model.predict(total_sqft_val)


# In[69]:


math.sqrt(((preds - sale_price_val) ** 2).mean())


# In[70]:


test_df = pd.read_csv('../input/test.csv')


# In[71]:


test_df.head()


# In[75]:


test_df['TotalSF'] = (
    test_df['GrLivArea'] +
    test_df['TotalBsmtSF'].fillna(0) +
    test_df['GarageArea'].fillna(0) +
    test_df['EnclosedPorch'].fillna(0) +
    test_df['ScreenPorch'].fillna(0))


# In[76]:


test_preds = model.predict(test_df[['TotalSF']])


# In[77]:


submission_df = pd.DataFrame(
    {'Id': test_df['Id'], 'SalePrice': np.exp(test_preds)}
)


# In[78]:


submission_df.head()


# In[79]:


submission_df.to_csv('my_sub.csv', index=False)


# In[81]:


FileLink('my_sub.csv')


# In[ ]:




