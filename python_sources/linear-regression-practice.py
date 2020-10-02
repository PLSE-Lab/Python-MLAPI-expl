#!/usr/bin/env python
# coding: utf-8

# **Note**
# This notebook is a practice from the Udemy course by Jose Portilla - Python for datascience and machine learning

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
get_ipython().run_line_magic('matplotlib', 'inline')
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/USA_Housing.csv')
df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.columns


# In[ ]:


sns.pairplot(df)


# In[ ]:


sns.distplot(df['Price'])


# In[ ]:


sns.heatmap(df.corr(),annot=True)


# In[ ]:


df.columns


# In[ ]:


X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]


# In[ ]:


y= df['Price']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state=101)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lm = LinearRegression()


# In[ ]:


lm.fit(X_train,  y_train)


# In[ ]:


print(lm.intercept_)


# In[ ]:


lm.coef_


# In[ ]:


X_train.columns


# In[ ]:


pd.DataFrame(lm.coef_, X.columns, columns=['Coeff'])


# In[ ]:


from sklearn.datasets import load_boston


# In[ ]:


boston = load_boston()


# In[ ]:


boston.keys()


# In[ ]:


print(boston['DESCR'])


# In[ ]:


predictions = lm.predict(X_test)


# In[ ]:


predictions


# In[ ]:


y_test


# In[ ]:


plt.scatter(y_test,predictions)


# In[ ]:


sns.distplot(y_test-predictions)


# **> Normal distributed residuals is a good sign**

# **A note on errors**
# * MAE = Mean Absolute Error it's the average error
# * MSE = Mean Squared Error punishes larger errors, which tends to be more useful in the real world
# * RMSE = Root Mean Squared Error is interpretable in the y units
# There are all loss functions because we want to minimize them

# In[ ]:


from sklearn import metrics


# In[ ]:


metrics.mean_absolute_error(y_test, predictions)


# In[ ]:


metrics.mean_squared_error(y_test, predictions)


# In[ ]:


np.sqrt(metrics.mean_squared_error(y_test, predictions))


# In[ ]:




