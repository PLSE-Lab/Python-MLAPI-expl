#!/usr/bin/env python
# coding: utf-8

# # House Prices for Beginners - Part 3 (advanced feature engineering, cross-validation, regularization and hyperparameter tuning)

# In[213]:


import math
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# ## Prepare features
# 
# The feature engineering is the same as performed in the first 2 notebooks

# In[214]:


df_train_orig = pd.read_csv('../input/train.csv')


# In[215]:


df_train = df_train_orig.copy()

df_train['TotalSF'] = df_train['GrLivArea'] + df_train['TotalBsmtSF'] + df_train['GarageArea'] + df_train['EnclosedPorch'] + df_train['ScreenPorch']
df_train['SalePrice'] = np.log(df_train['SalePrice'])

df_train['ExterQual'] = df_train.ExterQual.astype('category')
df_train['ExterQual'].cat.set_categories(['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered=True, inplace=True)
df_train['ExterQual'] = df_train['ExterQual'].cat.codes

df_train['Neighborhood'] = df_train['Neighborhood'].astype('category')
dummies = pd.get_dummies(df_train['Neighborhood'])
train_df_concat = pd.concat([df_train[['TotalSF', 'OverallQual', 'ExterQual']], dummies], axis=1)


# ## Intro to Cross Validation

# When the dataset is quite small, you can often find that the validation accuracy can vary a fair, depending on what range of feature values make it into the validation set vs the training set.
# 
# We can see that in action by setting the random state param to `train_test_split` to a few different values and observing the validation accuracy.

# In[216]:


for i in range(0, 5):
    train_df, train_val, sale_price_train, sale_price_val = train_test_split(
        train_df_concat, df_train['SalePrice'], test_size=0.2, random_state=i)

    model = LinearRegression()
    model.fit(train_df, sale_price_train)

    preds = model.predict(train_val)
    print(f'Val accuracy for iter {i}: {math.sqrt(((preds - sale_price_val)**2).mean())}')


# What we may like to do instead is to take a bunch of train/val splits and average the performance across each. That's where cross validation comes in. It separates the data into chunks of say, 4, then allocates one chunk to the val set then trains a model on the remaining chunks. It does this for all different chunks then returns a set of scores for each split.

# <img src="http://s5047.pcdn.co/wp-content/uploads/2015/06/07_cross_validation_diagram.png" width=200>
# 
# From http://blog.kaggle.com/2015/06/29/scikit-learn-video-7-optimizing-your-model-with-cross-validation/
# 
# Scikit learn provides an easy way to do cross validation using the `cross_val_score` function. You want to pass your model to the function, then the entire dataset.
# 
# Note that we're passing the `neg_mean_squared_error` to the scoring param as our scoring metric, so we'll need to take the negative of that, then the square root, to get our scores in the format we expect.

# In[217]:


model = LinearRegression()

scores = cross_val_score(model, train_df_concat,  df_train['SalePrice'], scoring='neg_mean_squared_error', cv=5)
scores = np.sqrt(-scores)


# We now have 5 different val scores calculated on different training / val splits.

# In[218]:


scores


# We can take the mean of those scores to get a better indication of our actual validation performance.

# In[219]:


scores.mean()


# ## Dealing with skewed features
# 
# Some features in the dataset have a very odd distribution, for example, the `LotArea` appears to have a huge right skew, we can see that in the histogram.

# In[220]:


plt.hist(df_train_orig['LotArea'][df_train_orig['LotArea'] > 0], bins=30)
plt.show()


# ![](http://)Trying to fit a single coefficient that can approximate that spread is going to be near difficult. If we take the log of a distribution, it will return something that approximates a normal distribution, as you can see below:

# In[221]:


df_train_orig['LotAreaLog'] = np.log1p(df_train_orig['LotArea'])
plt.hist(df_train_orig['LotAreaLog'][df_train_orig['LotAreaLog'] > 0], bins=30)
plt.show()


# We can now train a model with the original value and the new value to confirm that's helped.

# In[222]:


train_df_with_lot_val = pd.concat([train_df_concat, df_train_orig[['LotArea']]], axis=1)
train_df_with_log_lot_val = pd.concat([train_df_concat, df_train_orig[['LotAreaLog']]], axis=1)

model = LinearRegression()
scores_lot_val = np.sqrt(
    -cross_val_score(model, train_df_with_lot_val,  df_train['SalePrice'], scoring='neg_mean_squared_error', cv=5))
scores_log_lot_val = np.sqrt(
    -cross_val_score(model, train_df_with_log_lot_val,  df_train['SalePrice'], scoring='neg_mean_squared_error', cv=5))

print(f'Scores with LotArea {scores_lot_val.mean()}')
print(f'Scores with log(LotArea) {scores_log_lot_val.mean()}')


# ## Regularisation
# 
# Looking at the scores returned from `scores_log_lot_val`, it's clear that there's a reasonable spread in values:

# In[223]:


scores_log_lot_val


# And as you add more and more features, you might notice that that spread becomes more and more pronounced. You might even find some crazy high scores.
# 
# What's likely to be happening here, is that our model has learned some really high coefficient for a sparse column: it's placed huge importance on a feature it has only seen a couple of times and that doesn't generalise to the greater dataset.
# 
# One way we can help our model to learn coefficients that don't grow out of control is to add the magnitude of our coefficients to our error metric.
# 
# If our error metric is currently something like:
# 
# `sum((preds - error)**2)`
# 
# What if we added the sum of the absolute coefficient values to the error metric?
# 
# `sum((preds - error)**2) + sum(abs(coefs))`
# 
# Our model would now be trying to a) find the coefficients that minimise the loss but also b) find coefficients that are small.
# 
# Adding the absolute value of the coefficient magnitude is called an **l1_penalty**, we can also add the squared sum of the coefficients like so:
# 
# `sum((preds - error)**2) + sum(coefs**2)`
# 
# That's called **l2_penalty**
# 
# We may want to tune how much **l1** or **l2** penalty we apply. We do that by introducing an `alpha` param sets how important the penaltys are. Let's do that:
# 
# ```
# alpha = 0.1
# sum((preds - error)**2) + alpha * sum(coefs**2)
# ```
# 
# This is our first introduction to hyperparameters. We'll look at them in detail shortly.
# 
# LinearRegression with an **l1** penalty is called Lasso and with **l2** it's called Ridge. There's also **ElasticNet** which is a combination of both, with a hyperparam for how much **l1** vs **l2**. Let's start with Ridge.

# In[171]:


model = Ridge(alpha=0.01, max_iter=20000)
scores = np.sqrt(-cross_val_score(model, train_df_with_log_lot_val,  df_train['SalePrice'], scoring='neg_mean_squared_error', cv=5))
scores.mean()


# It appears to have helped ever so slightly, but you may find regularisation becomes more and more important as you add more features, or increase the complexity of your model.
# 
# The last question to ask is: how can you choose a good hyperparam value? For that, you may want to consider grid search.

# ## Hyperparameter tuning with GridSearch
# 
# Grid search let's us search over a space of hyperparameters looking for the best cross validation accuracy. We pass in a model and a dictionary of arguments with each value being a list of hyperparameters to try.
# 
# It will return the best parameter combination and the best validation accuracy in `grid.best_params_` and `grid.best_score_` respectively.
# 
# Let's try that.

# In[173]:


model = ElasticNet(max_iter=20000)

grid = GridSearchCV(model, {
    'alpha': [1, 0.1, 0.01, 0.04, 0.001, 0.0001],
    'l1_ratio': [0.0001, 0.001, 0.01, 0.5]
}, cv=5, scoring='neg_mean_squared_error', return_train_score=True, n_jobs=-1)

grid.fit(train_df_with_log_lot_val, df_train['SalePrice'])


# In[174]:


grid.best_params_


# In[175]:


math.sqrt(-grid.best_score_)


# Though the performance seems to be a little worse, we still have an optimal set of hyperparameters we can use.

# In[185]:


model = ElasticNet(alpha=0.0001, l1_ratio=0.5, max_iter=20000)
scores = np.sqrt(-cross_val_score(model, train_df_with_log_lot_val,  df_train['SalePrice'], scoring='neg_mean_squared_error', cv=5))

scores.mean()


# A slight improvement!

# ## Train on our whole dataset
# 
# One way to improve our model's test performance is to train it on the whole dataset. We won't be able to evaluate it before submitting our predictions, but we should expect slightly better performance on the leaderboard. Let's do that and submit.

# In[189]:


model = ElasticNet(alpha=0.0001, l1_ratio=0.5, max_iter=20000)
model.fit(train_df_with_log_lot_val, df_train['SalePrice'])


# ## Prepare test and submit predictions

# In[207]:


test_df = pd.read_csv('../input/test.csv')

test_df['TotalSF'] = test_df['GrLivArea'] + test_df['TotalBsmtSF'].fillna(0) + test_df['GarageArea'].fillna(0) + test_df['EnclosedPorch'].fillna(0) + test_df['ScreenPorch'].fillna(0)

test_df['ExterQual'] = test_df.ExterQual.astype('category')
test_df['ExterQual'].cat.set_categories(
    ['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered=True, inplace=True
)
test_df['ExterQual'] = test_df['ExterQual'].cat.codes

test_dummies = pd.get_dummies(test_df['Neighborhood'])
test_df_concat = pd.concat([test_df[['TotalSF', 'OverallQual', 'ExterQual']], test_dummies], axis=1)

test_df['LotAreaLog'] = np.log1p(test_df['LotArea'])
test_df_concat = pd.concat([test_df_concat, test_df[['LotAreaLog']]], axis=1)


# In[208]:


test_preds = model.predict(test_df_concat)


# In[210]:


pd.DataFrame(
    {'Id': test_df['Id'], 'SalePrice': np.exp(test_preds)}).to_csv('elasticnet.csv', index=False)


# In[ ]:




