#!/usr/bin/env python
# coding: utf-8

# # Overview:
# 
# This is a very quick and simple way to build the prediction model for the housing prices problem.
# 
# To build our model, we will drop every every column holding non-numeric values and fill in the remaining missing values with the means of each column.
# 
# Finally, we will use the XGBRegressor (without any parameter tuning) on the entire training set to train our model and use it for our predictions.

# ## Imports:

# In[ ]:


import pandas as pd
from xgboost import XGBRegressor


# ## Load Data:

# In[ ]:


train = pd.read_csv('../input/train.csv', index_col=0)
test = pd.read_csv('../input/test.csv', index_col=0)


# ## Preprocess the DataFrames:

# In[ ]:


#These two DataFrames will be used to train the model.
train_X = train.loc[:,:'SaleCondition'] #'SaleCondition' is the second last column before 'SalePrice'
train_y = train.loc[:,'SalePrice']

#This DataFrame will be used for the predictions
#we will submit.
test_X = test.copy()


# Next, we create a list holding the names of the numeric columns in the 'train_X' DataFrame:

# In[ ]:


numeric_cols = train_X.dtypes[train_X.dtypes != 'object'].index


# We will use this list to build our new `X` DataFrames that will hold only numeric data.

# In[ ]:


train_X = train_X[numeric_cols]

test_X = test_X[numeric_cols]


# Now that our two DataFrames contain only numeric data, we can go on and fill in the remaining missing values for each column using each column's mean.

# In[ ]:


train_X = train_X.fillna(train_X.mean())

test_X = test_X.fillna(test_X.mean())


# ## Build the Model:

# Now that our DataFrames are fit for processing, we can fit our model.
# 
# In this Notebook, we will use the XBGRegressor model.

# In[ ]:


model = XGBRegressor()

model.fit(train_X, train_y, verbose=False)


# ## Make the predictions:

# With our model fit and our test data (test_X) in proper processing format, we may now make our predictions.

# In[ ]:


predictions = model.predict(test_X)


# ## Submission:

# In[ ]:


#Create a DataFrame for our submission
submission = pd.DataFrame({'Id':test_X.index, 'SalePrice':predictions})


# In[ ]:


#Write the submission to a 'csv' file
submission.to_csv('submission.csv', index=False)


# ### Endnote:

# I'm not expecting amazing results with this submission.
# I just thought that it may be a good idea to make one of my very first competitions as simple as possible before I get deeper into 'feature-engineering'.
