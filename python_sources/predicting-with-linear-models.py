#!/usr/bin/env python
# coding: utf-8

# In this notebook, we will explore some more things we can do about predicting credit card turnover.

# In[ ]:


import pandas as pd
import numpy as np


# Let's look at the data once more:

# In[ ]:


data = pd.read_csv('../input/credit_card_clients_split.csv')
print(data.shape)
data.head()


# # Estimating the quality of the baseline

# We start by creating some variables we are going to use in two models

# In[ ]:


data['target'] = np.log(1 + data.avg_monthly_turnover)
data['log_income']= np.log(1 + data.income)
data['positive_target'] = (data.avg_monthly_turnover > 0).astype(int)


# Select only the data with present target variable

# In[ ]:


train_data = data[data.avg_monthly_turnover.notnull()].copy()


# In[ ]:


from sklearn.model_selection import train_test_split


# Split it into train and test sets for better evaluation of prediction quality

# In[ ]:


train, test = train_test_split(train_data, test_size=0.5, random_state=1, shuffle=True)


# Create a table for fitting the model of positive turnover

# In[ ]:


train_pos = train[train.avg_monthly_turnover > 0]


# Here is the linear regression from the baseline notebook - for the positive part.
# 
# $$\mathbb{E}(\log(turnover)|turnover>0) \approx \alpha + \beta \log(income)$$

# In[ ]:


from sklearn.linear_model import LinearRegression
linr = LinearRegression().fit(train_pos[['log_income']], train_pos['target'])


# Here is the logistic regression for the zero-or-positive part. It is based on the binary ("dummy") variables, which are a just a way to put a categorical variable (`sales_channel_id`) into the formula
# $$P(turnover>0)\approx \frac{1}{1+\exp\left(-(\alpha+\sum_j \beta_j [channelid = c_j]\right)}$$

# In[ ]:


from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
channel_dummy = pd.get_dummies(train.sales_channel_id)
logr = LogisticRegression().fit(channel_dummy, train.positive_target)


# This is the function for transforming the dataset and applying both models to it. 

# In[ ]:


def predict(dataset):
    p_positive = logr.predict_proba(pd.get_dummies(dataset.sales_channel_id))[:, 1]
    log_positive = linr.predict(dataset[['log_income']])
    prediction = np.exp(p_positive * log_positive)
    return prediction


# Now we try to evaulate prediction quality for both models, using the same metric (root means squaret logarithmic error) as in Kaggle. 

# In[ ]:


from sklearn.metrics import mean_squared_log_error


# In[ ]:


print(mean_squared_log_error(train.avg_monthly_turnover, predict(train)) ** 0.5)
print(mean_squared_log_error(test.avg_monthly_turnover, predict(test)) ** 0.5)


# # Multivariate models
# 
# Here we try to increase prediction quality by using all the available variables for both models. 
# 
# All the categorical variables will be encoded in binary format. The output of the cell below lists the resulting set of numerical variables, usable for both linear and logistic regression

# In[ ]:


numeric_predictors = ['log_income', 'age']
categorical_predictors = ['education', 'sales_channel_id', 'wrk_rgn_code']
x_train = pd.get_dummies(train[categorical_predictors + numeric_predictors], columns=categorical_predictors)
x_train_columns = x_train.columns
print(x_train_columns)


# We write a function for transforming both train and test datasets into the same expanded form. 

# In[ ]:


def preprocess(dataset):
    x = pd.get_dummies(dataset[categorical_predictors + numeric_predictors], columns=categorical_predictors)
    x = x.reindex(x_train_columns, axis=1).fillna(0)
    return x
x_train = preprocess(train)
print(x_train.shape)


# In[ ]:


from sklearn.linear_model import Ridge


# Because now there are a lot of variables, we will make models more stable by applying regularization to estimated coefficients - that is, assuming their prior distribution $\mathcal{N}(0,1)$.  In `scikit-learn`, the class `Ridge` does this for linear regression. 

# In[ ]:


level1 = Ridge(alpha=1)
level1.fit(x_train[train.positive_target == 1], train.target[train.positive_target==1])


# For logistic regression, amount of regularization is controlled with coefficient `C`

# In[ ]:


level0 = LogisticRegression(C=1, solver='liblinear')
level0.fit(x_train, train.positive_target)


# In[ ]:


def predict(dataset):
    x = preprocess(dataset)
    p_positive = level0.predict_proba(x)[:, 1]
    log_positive = level1.predict(x)
    prediction = np.exp(p_positive * log_positive)
    return prediction


# Wow! the error has decreased a lot!

# In[ ]:


print(mean_squared_log_error(train.avg_monthly_turnover, predict(train)) ** 0.5)
print(mean_squared_log_error(test.avg_monthly_turnover, predict(test)) ** 0.5)


# # Adding interactions between the features
# Here we are going to improve the model even more, by including pairwise products of all features - for the case if their combinations affect the target in an interesting way. 
# 
# Number of features grows a lot: from 77 to 1235 variables. Without regularization, our models would probably break now. 

# In[ ]:


def preprocess(dataset):
    x = pd.get_dummies(dataset[categorical_predictors + numeric_predictors], columns=categorical_predictors)
    x = x.reindex(x_train_columns, axis=1).fillna(0)
    # adding cross_products of features
    for i, c1 in enumerate(x_train_columns):
        for j, c2 in enumerate(x_train_columns):
            if j > i and c1[:10] != c2[:10]:
                x[c1 + '_' + c2] = x[c1] * x[c2]
    return x

x_train = preprocess(train)
print(x_train.shape)


# In[ ]:


level1 = Ridge(alpha=1)
level1.fit(x_train[train.positive_target == 1], train.target[train.positive_target==1])


# In[ ]:


level0 = LogisticRegression(C=1, solver='liblinear')
level0.fit(x_train, train.positive_target)


# In[ ]:


def predict(dataset):
    x = preprocess(dataset)
    p_positive = level0.predict_proba(x)[:, 1]
    log_positive = level1.predict(x)
    prediction = np.exp(p_positive * log_positive)
    return prediction


# The quality has improved even more, but not by much - most of the work has been already done by the simpler regerssions. 

# In[ ]:


print(mean_squared_log_error(train.avg_monthly_turnover, predict(train)) ** 0.5)
print(mean_squared_log_error(test.avg_monthly_turnover, predict(test)) ** 0.5)


# In[ ]:




