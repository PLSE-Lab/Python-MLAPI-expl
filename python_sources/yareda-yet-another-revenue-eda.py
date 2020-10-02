#!/usr/bin/env python
# coding: utf-8

# ## Overview
# 
# The purpose of this kernel is to take a look at the data, come up with some insights, and attempt to create a predictive model or two. This notebook is still **very** raw. I will work on it as my very limited time permits, and hope to expend it in the upcoming days and weeks.
# 
# ## Packages
# 
# First, let's load a few useful Python packages. This section will keep growing in subsequent versions of this EDA.

# In[ ]:


import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import time
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# Now let us look at the input folder. Here we find all the relevant files for this competition.

# In[ ]:


print(os.listdir("../input"))


# We see that the input folder only contains three files ```train.csv```, ```test.csv```, and ```sample_submission.csv```. It seems that for this competition we don't have to do any complicated combination and mergers of files.
# 
# Now let's import and take a glimpse at these files.

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
train_df.head()


# In[ ]:


train_df.shape


# In[ ]:


train_df.info()


# In[ ]:


train_df.isnull().values.any()


# A few things immediately stand out:
# 
# 1. There is almost a million datapoints, with 12 raw "features"
# 2. The most interesting features seem to be in JSON format. We will need to do something about them to get them into a format that is suitable for modeling.
# 3. There don't seem to be any missing values in the dataset, at least as far as the core raw features are concerned. Once we process JSON, it is possible that we'll find some missing values. Stay tuned.
# 
# Now let's take a look at the test dataset.

# In[ ]:


test_df = pd.read_csv('../input/test.csv')
test_df.head()


# In[ ]:


test_df.shape


# In[ ]:


test_df.info()


# In[ ]:


test_df.isnull().values.any()


# A few things about the test set that we notice:
# 
# 1. There is a close match in terms of the size with the train set (800,000 vs 900,000)
# 2. It also contains the JSON values features.
# 3. There don't seem to be any missing values in the dataset either.
# 
# Now let's reload the data, dealing with the JSON fields. For this we'll use [Julian's kernel function](https://www.kaggle.com/julian3833/1-quick-start-read-csv-and-flatten-json-fields/notebook):
# 

# In[ ]:


def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_df = load_df()\ntest_df = load_df("../input/test.csv")')


# Let's now look at the structure of the flattened files:

# In[ ]:


train_df.head()


# In[ ]:


train_df.info()


# In[ ]:


train_df.isnull().values.any()


# In[ ]:


train_df_describe = train_df.describe()
train_df_describe


# And now we'll look at the structure of the flattened test files:

# In[ ]:


test_df.head()


# In[ ]:


test_df.shape


# In[ ]:


test_df.info()


# In[ ]:


test_df_describe = test_df.describe()
test_df_describe


# The simplest "model" that we could build is to predict the mean value fro all test entries:

# In[ ]:


sample_submission = pd.read_csv('../input/sample_submission.csv')
sample_submission.head()


# In[ ]:


sample_submission['PredictedLogRevenue'] = np.log(1.26)
sample_submission.to_csv('simple_mean.csv', index=False)


# In[ ]:


sample_submission.head()


# In[ ]:




