#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# > ****LET'S CHECK OUR DATA

# In[ ]:


usa_housing =pd.read_csv("../input/USA_Housing.csv")
usa_housing.head()


# In[ ]:


usa_housing.info()


# In[ ]:


usa_housing.describe()


# In[ ]:


usa_housing.columns


# > **LET'S CREATE SOME PLOTS**

# In[ ]:


sns.pairplot(usa_housing)


# In[ ]:


sns.distplot(usa_housing['Price'])


# 

# In[ ]:


sns.heatmap(usa_housing.corr(),annot = True)


# 
# > ****Training a Linear Regression Model 
# > Let's now begin to train out regression model! We will need to first split up our data into an X array that contains the features to train on, and a y array with the target variable, in this case the Price column. We will remove the Address column because it only has text info that the linear regression model can't use.
# 
# X and y arrays

# In[ ]:


X = usa_housing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = usa_housing['Price']


# ##Train Test Split
# Now let's split the data into a training set and a testing set. We will train out model on the training set and then use the test set to evaluate the model.

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state =101)


# Creating and Training the Model

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lm = LinearRegression()


# In[ ]:


lm.fit(X_train,y_train)


# Model Evaluation
# 
# Let's evaluate the model by checking out it's coefficients and how we can interpret them.

# In[ ]:


# print the intercept
print(lm.intercept_)


# In[ ]:


print(X.columns)


# In[ ]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# Predictions from our Model
# 
# Let's grab predictions off our test set and see how well it did!

# In[ ]:


predictions = lm.predict(X_test)


# In[ ]:


plt.scatter(y_test,predictions)


# Residual Histogram

# In[ ]:


sns.distplot((y_test-predictions),bins=50);


# In[ ]:


from sklearn import metrics


# In[54]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




