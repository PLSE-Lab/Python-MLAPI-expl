#!/usr/bin/env python
# coding: utf-8

# # *Using H2o AutoML to predict housing prices*

# This quick tutorial shows how you can use H2o AutoML to train and evaluate a large amount of models with only a few lines of coding.
# Please, upvote if you find it useful.

# ### Load libraries

# In[ ]:


# You can easily install the library using pip
get_ipython().system('pip install h2o')


# In[ ]:


# And then load the libraries you'll use in this notebook
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


import h2o
from h2o.automl import H2OAutoML


# In[ ]:


# Initialize your cluster
h2o.init()


# ### Load train and test datasets

# In[ ]:


train = h2o.import_file('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = h2o.import_file('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.head()


# ### EDA, Data Processing and Feature Engineering

# I won't perform neither EDA nor Feature Engineering as these are not the focus of this kernel. At least not an extensive one. Here I'll only drop registers whose `GrLivArea` are above 4500 and log transform the target variable `SalePrice`.

# In[ ]:


train = train[train['GrLivArea'] < 4500]
train['SalePrice'] = train['SalePrice'].log1p()


# ### Using H2o AutoML

# The H2OAutoML function is quite easy to use. You specify the dataset you will use for training at `training_frame`, while `x` and `y` receives the column names of the features which will be used and the name of the target variable, respectively.
# 
# You can customize your AutoML to fit your needs. You can add or exclude algorithms, set nfolds for cross-validation, choose metrics, use validation sets, early stopping and so on. Please check the documentation at http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html

# In[ ]:


# Identify predictors and response
x = [col for col in train.columns if col not in ['Id','SalePrice']]
y = 'SalePrice'
test_id = test['Id']


# Let's now create our model:

# In[ ]:


aml = H2OAutoML(max_runtime_secs=90000, seed = 99, stopping_metric = 'RMSLE')
aml.train(x = x, y = y, training_frame = train)


# The Leaderboard displays the performance of the trained models

# In[ ]:


lb = aml.leaderboard; lb


# The attribute `leader` of your AutoML object holds the data about your best model.

# In[ ]:


aml.leader


# It also gives you the Feature Importance

# In[ ]:


aml.leader.varimp_plot()


# ### Predict

# You can simply call predict on your best model

# In[ ]:


preds = aml.leader.predict(test)


# In[ ]:


# Convert results back(they had been transformed using log, remember?) and save them in a csv format.
result = preds.expm1()
sub = test_id.cbind(result)
sub.columns = ['Id','SalePrice']
sub = sub.as_data_frame()
sub.to_csv('submission.csv', index = False)


# In[ ]:


import pandas as pd
sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

