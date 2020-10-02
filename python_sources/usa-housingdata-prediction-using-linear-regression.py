#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# *Check out the Data*

# In[ ]:


USAhousing = pd.read_csv('../input/USA_Housing.csv')


# In[ ]:


USAhousing.info()


# In[ ]:


USAhousing.head()


# In[ ]:


USAhousing.describe()


# In[ ]:


USAhousing.columns


# 

# **EDA**
# * Let's create some simple plots to check out the data!

# In[ ]:


sns.pairplot(USAhousing)


# In[ ]:


sns.distplot(USAhousing['Price'])


# In[ ]:


sns.barplot


# In[ ]:


sns.heatmap(USAhousing.corr())


# **Training a Linear Regression Model**

# Let's now begin to train out regression model! We will need to first split up our 
# data into an X array that contains the features to train on, and a y array with the target 
# variable, in this case the Price column. We will toss out the Address column because it only
# has text info that the linear regression model can't use.

# **X and y arrays**

# In[ ]:


X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']


# **Train Test Split**

# Now let's split the data into a training set and a testing set. We will train out model on the training set and then use the test set to evaluate the model.

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# **Creating and Training the Model**

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lm = LinearRegression()


# In[ ]:


lm.fit(X_train,y_train)


# **Model Evaluation**

# Let's evaluate the model by checking out it's coefficients and how we can interpret them.

# In[ ]:


# print the intercept
print(lm.intercept_)


# In[ ]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[ ]:


Interpreting the coefficients:
    


# * Holding all other features fixed, a 1 unit increase in Avg. Area Income is associated with an increase of $21.52.
# * Holding all other features fixed, a 1 unit increase in Avg. Area House Age is associated with an increase of $164883.28.
# * Holding all other features fixed, a 1 unit increase in Avg. Area Number of Rooms is associated with an increase of $122368.67.
# * Holding all other features fixed, a 1 unit increase in Avg. Area Number of Bedrooms is associated with an increase of $2233.80 .
# * Holding all other features fixed, a 1 unit increase in Area Population is associated with an increase of $15.15.
# 
# 

# **Predictions from our Model**

# Let's grab predictions off our test set and see how well it did!

# In[ ]:


predictions = lm.predict(X_test)


# In[ ]:


plt.scatter(y_test,predictions)


# **Residual Histogram**

# In[ ]:


sns.distplot((y_test-predictions),bins=50);


# ## Regression Evaluation Metrics
# 
# 
# Here are three common evaluation metrics for regression problems:
# 
# **Mean Absolute Error** (MAE) is the mean of the absolute value of the errors:
# 
# $$\frac 1n\sum_{i=1}^n|y_i-\hat{y}_i|$$
# 
# **Mean Squared Error** (MSE) is the mean of the squared errors:
# 
# $$\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2$$
# 
# **Root Mean Squared Error** (RMSE) is the square root of the mean of the squared errors:
# 
# $$\sqrt{\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2}$$
# 
# Comparing these metrics:
# 
# - **MAE** is the easiest to understand, because it's the average error.
# - **MSE** is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
# - **RMSE** is even more popular than MSE, because RMSE is interpretable in the "y" units.
# 
# All of these are **loss functions**, because we want to minimize them.

# In[ ]:


from sklearn import metrics


# In[ ]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:




