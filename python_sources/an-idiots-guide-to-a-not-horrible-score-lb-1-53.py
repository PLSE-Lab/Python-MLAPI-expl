#!/usr/bin/env python
# coding: utf-8

# # An idiots guide to a not horrible score.  Forget exploratory data analysis, forget parameter tuning, just get something that works sort of well.  Let's do it in less than 30 lines of code.

# Load some packages and the data.  Create a numpy array called X with the training set features and a numpy array called y with the target values.  Create a numpy array with the test set features.  Let's log (base e) transform the target variable since it is vast in scale.

# In[1]:


import pandas as pd
import numpy as np

dataset = pd.read_csv('../input/train.csv')
testset = pd.read_csv('../input/test.csv')

X = dataset.iloc[:, 2:].values
y = dataset.iloc[:, 1].values
X_Test = testset.iloc[:,1:].values
y = np.log(y)


# Let's forget all about EDA.  Other people showed that there are no missing values, everything is numeric, and no wild outliers.  Let's just get rid of any features with no variance.  This will include variables with all the same value.

# In[2]:


from sklearn.feature_selection import VarianceThreshold
feature_selector = VarianceThreshold()
X = feature_selector.fit_transform(X)
X_Test = feature_selector.transform(X_Test)


# Now let's get to making a regressor.  A good place to start is XGBoost.  Let's just make most of the paramters at their defaults.  Maybe use 300 estimators.  Instantiate regressor, fit model, bada boom, bada bing.

# In[3]:


from xgboost import XGBRegressor
regressor = XGBRegressor(n_estimators=300)
regressor.fit(X, y)


# Okee dokee.  Now let's make predictions on the test data with our not-so-fancy model.  Don't forget to exponentiate them since the target values were log transformed.

# In[4]:


results = regressor.predict(X_Test)
results = np.exp(results)


# Boooooooya!  All thats left to do is write the results to a file and submit it.  Don't ya' just love pandas?

# In[8]:


submission = pd.DataFrame()
submission['ID'] = testset['ID']
submission['target'] = results
submission.to_csv('submission.csv', index=False)


# Hey, that's 21 lines of code.  Not so bad.
