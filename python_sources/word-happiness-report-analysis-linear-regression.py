#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[55]:


data = pd.read_csv('../input/world-happiness-report-2019.csv')


# In[56]:


data.head()


# In[57]:


data.describe()


# In[58]:


data.isnull().sum()


# In[59]:


data.columns


# In[60]:


from sklearn.preprocessing import Imputer


# In[61]:


imputer = Imputer(missing_values='NaN', strategy='median', axis=0)


# In[62]:


data.iloc[:, 3:11] = imputer.fit_transform(data.iloc[:, 3:11])


# In[63]:


data.isnull().sum()


# In[64]:


fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(data.corr(),ax=ax,annot=True,linewidth=0.05,fmt='.2f')


# In[65]:


data.plot(x='Positive affect', y='Negative affect', kind='scatter')
plt.xlabel('Positive affect')
plt.ylabel('Negative affect')


# In[66]:


data.plot(x='Positive affect', y='Freedom', kind='scatter')
plt.xlabel('Positive affect')
plt.ylabel('Freedom')


# In[67]:


data.plot(x='Positive affect', y='Generosity', kind='scatter')
plt.xlabel('Positive affect')
plt.ylabel('Generosity')


# In[68]:


data.plot(x='Positive affect', y='Corruption', kind='scatter')
plt.xlabel('Positive affect')
plt.ylabel('Corruption')


# In[69]:


data.plot(x='Ladder', y='SD of Ladder', kind='scatter')
plt.xlabel('Ladder')
plt.ylabel('SD of Ladder')


# In[70]:


X = data.drop(['Country (region)', 'Healthy life\nexpectancy'], axis=1)
y = data['Healthy life\nexpectancy']


# In[71]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# In[72]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[73]:


linreg = LinearRegression()
linreg.fit(X_train, y_train)


# In[74]:


y_pred = linreg.predict(X_test)
print('r2 score: ', r2_score(y_test, y_pred))


# In[ ]:




