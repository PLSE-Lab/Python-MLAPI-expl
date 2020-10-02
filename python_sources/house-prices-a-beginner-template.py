#!/usr/bin/env python
# coding: utf-8

# # Introduction
# **Please feel free to use this notebook as a template for your project**. 
# 
# Goal:
# > It is your job to predict the sales price for each house. For each Id in the test set, you must predict the value of the SalePrice variable. 
# 
# 
# 

# In[ ]:


# add necessary pckgs here:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Loading data into dataframes
train_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv") # loading train data 
test_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv") # loading test data


# # EDA
# 
# Some things you could look at:
# 
# * Data exploration
# * Data cleaning
# * Outlier Detection
# * Feature Engineering
# 
# ... etc

# In[ ]:


# (Quick) Example: 

# Print out shape of data (rows, coloumns) for train and test
print('Shape of train data: ',train_df.shape, '\nShape of test data: ' ,test_df.shape)

# Summary stats
print(train_df.describe())


# corelation plot
corrmat = train_df.corr()
top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.6]
plt.figure(figsize=(6,6))
g = sns.heatmap(
    train_df[top_corr_features].corr(), 
    annot = True, cmap = "Blues", 
    cbar = False, vmin = .5, 
    vmax = .7, square=True
    )


# In[ ]:


# NOTE: GrLivArea and OverallQual is strongly correlated to SalePrice. I'll use these two features for my model down below. 
#-----------------------------
# Your EDA code goes here


# # Modeling
# 
# Try a few models. I'll showcase a simple model to kick off. - inspired by Dan S. Becker: https://www.kaggle.com/dansbecker/submitting-from-a-kernel

# ## RandomForestRegressor

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# pull data into target (y) and predictors (X)
y = train_df.SalePrice
pred_cols = ['OverallQual', 'GrLivArea']

# Create training predictors data
X = train_df[pred_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.3, random_state=0)

# Fit model
my_model = RandomForestRegressor()
my_model.fit(X_train, y_train)

# Accuracy eval
print(my_model.score(X_test, y_test)) # - 0.7522067150061065 (around 75%) - decent, but can definitevly be improved (This is up to you to do)

# Predict
# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = test_df[pred_cols]
# Use the model to make predictions
predicted_prices = my_model.predict(test_X)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_prices)

 


# # Submission
# 
# Writes your predictions to a csv file.

# In[ ]:


my_submission = pd.DataFrame({'Id': test_df.Id, 'SalePrice': predicted_prices})
# you could use any filename. I went with submission here
my_submission.to_csv('submission.csv', index=False)

