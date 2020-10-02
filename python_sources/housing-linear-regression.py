#!/usr/bin/env python
# coding: utf-8

# In[65]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# In[66]:


df = pd.read_csv('../input/Housing.csv')
df.head()


# In[67]:


df.size


# ## Linear Regression One Variable

# In[68]:


df[['lotsize', 'price']].head()


# In[69]:


plt.scatter(df['lotsize'], df['price'], color='red')
plt.axis([pd.Series.min(df['lotsize']), pd.Series.max(df['lotsize']), pd.Series.min(df['price']), pd.Series.max(df['price'])])
plt.show()


# In[72]:


x = df[['lotsize']]
y = df[['price']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
y_test.head()


# In[73]:


x_test.head()


# In[75]:


model = linear_model.LinearRegression()
model.fit(x_train, y_train)


# In[76]:


price_pred = model.predict(x_test)


# In[77]:


print('Coefficients: \n', model.coef_)
print("Mean squared error: %.2f" 
      % mean_squared_error(y_test, price_pred))
print('Variance score: %.2f' % r2_score(y_test, price_pred))


# In[80]:


plt.scatter(x['lotsize'], y['price'], color='red')
plt.plot(x_test['lotsize'], price_pred, color='blue')
plt.show()


# ## Linear Regression Multiple Variables

# In[51]:


df.head()


# In[88]:


df = df.replace(to_replace='yes', value=1, regex=True)
df = df.replace(to_replace='no', value=0, regex=True)


# In[89]:


X = df[['lotsize','bedrooms','stories','bathrms','bathrms','driveway','recroom',
        'fullbase','gashw','airco','garagepl','prefarea']]
y = df[['price']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


# In[90]:


model = linear_model.LinearRegression()
model.fit(X_train, y_train)
predicted = model.predict(X_test)


# In[91]:


print('Coefficients: \n', model.coef_)
print("Mean squared error: %.2f" 
      % mean_squared_error(y_test, predicted))
print('Variance score: %.2f' % r2_score(y_test, predicted))


# In[ ]:




