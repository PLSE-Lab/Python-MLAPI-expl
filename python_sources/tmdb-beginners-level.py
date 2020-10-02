#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import  matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('../input/train.csv')


# We need the budget attribute and the popularity.
# Lets try linear regression with it.
# 

# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


X = df[['budget','popularity']]
X.head()


# In[ ]:


y = df.revenue
y.head()


# In[ ]:


from sklearn import linear_model
model = linear_model.LinearRegression()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(df.budget,df.revenue)


# In[ ]:


plt.scatter(df.popularity,df.revenue)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df[['budget']],df[['revenue']])


# In[ ]:


model.fit(X_train,y_train)


# In[ ]:


model.score(X_test,y_test)


# In[ ]:


test = pd.read_csv('../input/test.csv')
test.head()


# In[ ]:


features = ['budget']
target = 'revenue'


# In[ ]:


predictions = model.predict(test[features])
predictions


# In[ ]:





# In[ ]:


submission = pd.DataFrame()
submission['id'] = test['id']
submission['revenue'] = predictions
submission.to_csv('submission.csv', index=False)


# In[ ]:


df = pd.read_csv('submission.csv')
df.head()


# In[ ]:




