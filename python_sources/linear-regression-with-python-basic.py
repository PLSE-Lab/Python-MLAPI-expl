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
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
print(os.listdir("../input"))


# # Check out the Data

# In[ ]:


USA_Housing = pd.read_csv("../input/usa-housingcsv/USA_Housing.csv")
USA_Housing.head()


# In[ ]:


USA_Housing.info()


# In[ ]:


USA_Housing.describe()


# In[ ]:


USA_Housing.columns


# # Let's create some plots 

# In[ ]:


sns.pairplot(USA_Housing)


# In[ ]:


sns.distplot(USA_Housing['Price'])


# In[ ]:


sns.heatmap(USA_Housing.corr())


# # Training a Linear Regression Model
# Let's now begin to train out regression model! We will need to first split up our data into an X array that contains the features to train on, and a y array with the target variable, in this case the Price column. We will toss out the Address column because it only has text info that the linear regression model can't use.
# 
# ### X and y arrays

# In[ ]:


x = USA_Housing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USA_Housing['Price']


# # Train Test Split
# Now let's split the data into a training set and a testing set. We will train out model on the training set and then use the test set to evaluate the model.

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.4,random_state=101)


# # Creating and Training the Model

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lm = LinearRegression()


# In[ ]:


lm.fit(x_train, y_train)


# # Model Evaluation
# Let's evaluate the model by checking out it's coefficients and how we can interpret them

# In[ ]:


# print the intercept
print(lm.intercept_)


# In[ ]:


coeff_df = pd.DataFrame(lm.coef_,x.columns,columns=['Coefficient'])
coeff_df


# # Predictions from our Model
# Let's grab predictions off our test set and see how well it did!

# In[ ]:


predictions = lm.predict(x_test)


# In[ ]:


plt.scatter(y_test,predictions)


# ## Residual Histogram

# In[ ]:


sns.distplot(y_test-predictions, bins=50);


# # Regression Evaluation Metrics

# In[ ]:


from sklearn import metrics


# In[ ]:


print('MAE:',metrics.mean_absolute_error(y_test,predictions))
print('MSE:',metrics.mean_squared_error(y_test,predictions))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,predictions)))

