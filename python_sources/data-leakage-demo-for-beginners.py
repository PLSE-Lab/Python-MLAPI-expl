#!/usr/bin/env python
# coding: utf-8

# ## Data Leakage Demo for Beginners

# The term "data leakage" typically means a situation where you inadvertantly make a mistake which "leaks" information about the target into the feature space.  What will generally follow is that you will have a surprisingly good local validation score, but when you then make predictions on unseen data, your model will perform very poorly.  In this demo, I will demonstrate one way such a leak can happen with the Santander data.

# First, lets load some libraries, the data, and do the typical reformatting.

# In[ ]:


# import some libraries
import pandas as pd
import numpy as np

# load data
trainset = pd.read_csv('../input/train.csv')
testset = pd.read_csv('../input/test.csv')

# create target vector
y = trainset.iloc[:, 1].values

# drop some columns such as ID and target
trainset = trainset.iloc[:,2:]
testset = testset.iloc[:,1:]


# Now lets make numpy arrays to input into our LightGBM model

# In[ ]:


# create numpy arrays
X = trainset.iloc[:, :].values
X_Test = testset.iloc[:, :].values


# ## So now let's perform some dimensionality reduction method, such as Linear Discriminant Analysis

# ## <font color='red'>**Notice that this is where the data leakage occurs</font>

# In[ ]:


# Applying LDA for dimensionality reduction
# Here is where the data leakage occurs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA()
X = lda.fit_transform(X, y.astype(int))
X_Test = lda.transform(X_Test)


# By performing this tranformation, I am accidentally leaking information about the target into the feature space.

# Lets continue as normal.  Create our train/test split and prepare for LightGBM model training.

# In[ ]:


# create out train/test split and prepare for LightGBM model training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42)

import lightgbm as lgb
train = lgb.Dataset(X_train, label=np.log1p(y_train))
test = lgb.Dataset(X_test, label=np.log1p(y_test))


# Lets create a simple LightGBM regressor using basically the default parameters and see what we get.

# In[ ]:


# Train LightGBM model with simple default parameters
params = {
    'boosting_type': 'gbdt',
    'metric': 'rmse',
    'zero_as_missing':True
    }
regressor = lgb.train(params, 
                      train, 
                      3000, 
                      valid_sets=[test], 
                      early_stopping_rounds=100, 
                      verbose_eval=100)


# ## Wow!  A 0.84 RMSE score!  I am the Master!

# ## Until I make predictions on the unseen data and make a submission.  I get a terrible LB score.

# This is one demonstration of what is traditionally considered data leakage.  There are other forms too.  This is an extreme example, but I think it demonstrates the concept well for beginners.
