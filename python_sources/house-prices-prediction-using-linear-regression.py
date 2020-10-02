#!/usr/bin/env python
# coding: utf-8

# # House prices prediction using linear regression

# * Autor: Mateus Mendes Ramalho da Silva
# *        Bachelor in computer science
# *        Data scientist
# *        mateus.mendes.mmr@gmail.com
#        

# ### Importing libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ### Loading dataframes

# In[ ]:


train_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')


# In[ ]:


test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# ### Checking dataframe's informations

# Checking the head of both dataframes:

# In[ ]:


train_df.head()


# In[ ]:


train_df.info()


# In[ ]:


test_df.info()


# We can notice that the train dataframe has 80 columns, but i've decided to use only the int64 format data. So i'm going to choose only these columns of the dataframe for training my model

# In[ ]:


test_df = test_df.select_dtypes(include=['int64'])


# I'm going to check which are the remaining columns on the test dataframe, since we are only using these columns for prediction i'm going to choose the same columns on the train dataframe.

# First i'm also clean the NaN numeric data with the mean of its column.

# In[ ]:


train_df.fillna(value = train_df.mean())


# ### Making a heatmap of the correlation

# It's always interesting to make a data visualization for checking if there are a good relation between the data and if it fits in a linear regression.

# In[ ]:


plt.figure(figsize=(16,8))
sns.heatmap(train_df.corr())


# We can check that we have some high values of correlation in some atributes with the SalePrice, so we can conclude that the linear regression may fit.

# Let's check which are the remaining columns on the test dataframe for making a selection of the same columns in the train dataframe.

# In[ ]:


test_df.columns


# Selecting predictors:

# In[ ]:


X = train_df[['Id', 'MSSubClass', 'LotArea', 'OverallQual', 'OverallCond',
       'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
       'TotRmsAbvGrd', 'Fireplaces', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
       'MoSold', 'YrSold']]


# SalePrice it's what it's going to be predicted

# In[ ]:


y = train_df[['SalePrice']]


# As i am still a initiate, i decided that would also be interesting checking the data using the test data from the train dataframe, because in this way i can check the accuracy since i don't know many methods of checking it yet.

# Importing train_test_split for splitting the train data into test and train:

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# Building the simulation model:

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


simul_lm = LinearRegression()


# In[ ]:


simul_lm.fit(X_train, y_train)


# In[ ]:


simul_predict = simul_lm.predict(X_test)


# In[ ]:


plt.scatter(y_test, simul_predict)


# We can see that it fits good

# Checking some accuracy metrics:

# In[ ]:


from sklearn import metrics


# In[ ]:


print('MAE', metrics.mean_absolute_error(y_test,simul_predict))


# In[ ]:


print('MSE', metrics.mean_squared_error(y_test,simul_predict))


# In[ ]:


print('RMSE', np.sqrt(metrics.mean_squared_error(y_test,simul_predict)))


# ### Building the model:

# Importing library:

# In[ ]:


lm = LinearRegression()


# Fitting model:

# In[ ]:


lm.fit(X, y)


# Let's check some parameters:

# Question: Where the data cross the y axis?

# In[ ]:


print(lm.intercept_)


# Checking the coefficients:

# In[ ]:


print(lm.coef_)


# ### Making the predict:

# In[ ]:


predict = lm.predict(test_df)


# In[ ]:


print(predict)


# In[ ]:


results = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')


# ### Making the submission:

# In[ ]:


submission = pd.DataFrame()
submission['Id'] = results['Id']
submission['SalePrice'] = predict


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('submission.csv', index=False)

