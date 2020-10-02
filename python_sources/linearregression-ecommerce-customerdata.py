#!/usr/bin/env python
# coding: utf-8

# In[145]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
# Any results you write to the current directory are saved as output.


# In[146]:


customers = pd.read_csv('../input/Ecommerce Customers')


# In[147]:


customers.head(5)


# In[148]:


customers.describe()


# In[149]:


sns.jointplot(data=customers, x='Time on Website', y='Yearly Amount Spent')


# In[150]:


sns.jointplot(data=customers, x='Time on App', y='Yearly Amount Spent')


# In[151]:


sns.jointplot(data=customers, x='Time on App', y='Length of Membership')


# In[152]:


sns.pairplot(customers)


# In[153]:


plt.scatter(customers['Length of Membership'], customers['Yearly Amount Spent'])
plt.xlabel('Length of Membership')
plt.ylabel('Yearly Amount Spent')


# In[154]:


customers.columns


# In[156]:


Y = customers['Yearly Amount Spent']


# In[157]:


X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]


# In[158]:


X.columns.shape


# In[159]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)


# In[160]:


lm = LinearRegression()
lm.fit(X_train, Y_train)


# In[161]:


lm.coef_


# In[162]:


preds = lm.predict(X_test)


# In[163]:


plt.scatter(Y_test, preds)
plt.xlabel('Y true values')
plt.ylabel('Y predictions')


# In[164]:


print('MAE', metrics.mean_absolute_error(Y_test, preds))


# The reason that we used the Linear regression here is NOT to predict the values but to find the coefficients of the dataset
# and we used the test dataset to make sure this coefficients lead to good results and are valueable for comparison between 
# different columns are dataset.

# In[166]:


sns.distplot((Y_test-preds), bins=30)


# In[168]:


pd.DataFrame(lm.coef_, X.columns, columns=['Coeff'])


# This dataframe implies that if we hold all other values to a fixed value, how much increase/decrease we will see if we increase
# one of the coefficients by one.
# Therefore the company should focus the most on "Length of Membership"
# and between Time on website and Time on app , the mobile application shows more potential for growth.

# In[ ]:




