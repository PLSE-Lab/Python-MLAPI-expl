#!/usr/bin/env python
# coding: utf-8

# # Auto Linear Regression
# #### We have seen Auto ML like H2O which is a blackbox approach to generate models. 
# During our model building process, we try with brute force/TrialnError/several combinations to come up with best model. 
# However trying these possibilities manually is a laborious process.
# In order to overcome or atleast have a base model automatically I developed this auto linear regression using backward feature elimination technique.
# 
# The library/package can be found [here](https://pypi.org/project/kesh-utils/) and source code [here](https://github.com/KeshavShetty/ds/tree/master/KUtils/linear_regression)
# 

# # How Auto LR works?
# We throw the cleaned dataset to autolr.fit(<<parameters>>)
# The method will 
# - Treat categorical variable if applicable(dummy creation/One hot encoding)
# - First model - Run the RFE on dataset
# - For remaining features elimination - it follows backward elimination - one feature at a time
#     - combination of vif and p-values of coefficients (Eliminate with higher vif and p-value combination
#     - vif only (or eliminate one with higher vif)
#     - p-values only (or eliminate one with higher p-value)
# - Everytime when a feature is identified we build new model and repeat the process
# - on every iteration if adjusted R2 affected significantly, we re-add/retain it and select next possible feature to eliminate.
# - Repeat until program can't proceed further with above logic.
# 

# # Action time
# Lets try this library and see how it works
# 
# To demonstrate the library I used the one of the popular dataset [UCI Diamond dataset](https://www.kaggle.com/shivam2503/diamonds) from Kaggle
# 
# 

# In[59]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[60]:


# Install Auto Linear Regression (Part of KUtils Package)
get_ipython().system('pip install kesh-utils')


# In[61]:


# Load the custom packages from kesh-utils
from KUtils.eda import chartil
from KUtils.eda import data_preparation as dp


# In[62]:


# The library is auto_linear_regression we give alias as autolr
from KUtils.linear_regression import auto_linear_regression as autolr


# In[63]:


# Use 3 decimal places for decimal number (to avoid displaying as exponential format)
pd.options.display.float_format = '{:,.3f}'.format


# In[ ]:


import warnings  
warnings.filterwarnings('ignore')


# In[64]:


# Load the dataset
diamond_df = pd.read_csv('../input/diamonds.csv')


# In[65]:


# Have a quick look on the top few records of the dataset 
diamond_df.head()


# Price is target column and the first column seems to be just a sequence

# In[66]:


diamond_df.describe()


# In[67]:


# Drop first column which is just a sequence
diamond_df = diamond_df.drop(diamond_df.columns[0], axis=1)


# In[68]:


diamond_df.shape


# 53,920 rows, 10 features including target column 'price'

# In[69]:


# Null checks
diamond_df.isnull().sum()


# In[70]:


# Plot the nulls as barchart (Null count in each features)
dp.plotNullInColumns(diamond_df)


# In[71]:


# Number of unique values in each column (Check in both Train and Test for missing categorial label in any)
{x: len(diamond_df[x].unique()) for x in diamond_df.columns}


# In[72]:


# Plot unique values in each feature
dp.plotUnique(diamond_df, optional_settings={'sort_by_value':True})


# In[73]:


# Some EDA (Using kesh-utils chartil package)
chartil.plot(diamond_df, diamond_df.columns, optional_settings={'include_categorical':True, 'sort_by_column':'price'})


# In[74]:


# Have a quick look on different feature and their relation
chartil.plot(diamond_df, ['carat', 'depth','table', 'x','y','z','price', 'cut'], chart_type='pairplot', optional_settings={'group_by_last_column':True})


# In[75]:


chartil.plot(diamond_df, ['color', 'price'], chart_type='distplot')


# In[76]:


chartil.plot(diamond_df, ['clarity', 'price'])


# # Auto Linear Regression in Action
# The method <b><u>autolr.fit()</u></b> has below parameters
# - df, (The full dataframe)
# - dependent_column, (Target column)
# - p_value_cutoff = 0.01, (Threashold p-values of features to use while filtering features during backward elimination step, Default 0.01)
# - vif_cutoff = 5, (Threashold co-relation of vif values of features to use while filtering features during backward elimination step, Default 5)
# - acceptable_r2_change = 0.02, (Restrict degradtion of model efficiency by controlling loss of change in R2, Default 0.02)
# - scale_numerical = False, (Flag to convert/scale numerical fetures using StandardScaler)
# - include_target_column_from_scaling = True, (Flag to indiacte weather to include target column from scaling)
# - dummies_creation_drop_column_preference='dropFirst', (Available options dropFirst, dropMax, dropMin - While creating dummies which clum drop to convert to one hot)
# - train_split_size = 0.7, (Train/Test split ration to be used)
# - max_features_to_select = 0, (Set the number of features to be qualified from RFE before entring auto backward elimination)
# - random_state_to_use=100, (Self explanatory)
# - include_data_in_return = False, (Include the data generated/used in Auto LR which might have gobne thru scaling, dummy creation etc.)
# - verbose=False (Enable to print detailed debug messgaes)

# In[77]:


model_info = autolr.fit(diamond_df, 'price', 
                     scale_numerical=True, acceptable_r2_change = 0.005,
                     include_target_column_from_scaling=True, 
                     dummies_creation_drop_column_preference='dropFirst',
                     random_state_to_use=100, include_data_in_return=True, verbose=True)


# Above method returns 'model_info' dictionary which will have all the details used while performing auto fit. 
# Will go thru one by one

# In[78]:


# Print all iteration info
model_info['model_iteration_info'].head()


# In[79]:


# Final set of features used in final model 
model_info['features_in_final_model']


# In[80]:


# vif-values of features used in final model  
model_info['vif-values']


# In[81]:


# p-values of features used in final model  
model_info['p-values']


# In[82]:


# Complete stat summary of the OLS model
model_info['model_summary']


# Finally we have the model with respective coefficients. 

# * ### Go back to autolr.fit() method and try different cut-offs or finetune other parameter/options

# In[ ]:




