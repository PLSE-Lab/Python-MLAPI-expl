#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gc, os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import defaultdict
from tqdm import tnrange, tqdm_notebook
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import SGDClassifier
train_fpath = "../input/train.csv"
test_fpath = "../input/test.csv"


# ## Helper Functions

# In[2]:


sample_counts_on_each_day = {
    "2017-11-06": 9308568,
    "2017-11-07": 59633310,
    "2017-11-08": 62945075,
    "2017-11-09": 53016937
}
columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time', 'is_attributed']
dtypes = {
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8',
    'click_id': 'uint32'
}
def get_samples_by_date(at):
    nrows = 0
    skiprows = 1
    for date, count in sample_counts_on_each_day.items():
        if date == at:
            nrows = count
        elif date < at:
            skiprows += count
    return pd.read_csv(train_fpath, skiprows=skiprows, nrows=nrows, names=columns, dtype=dtypes, parse_dates=["click_time"])
def ratio(x):
    return x.shape[0] / sample_counts_on_each_day[sample_date]


# ## The ratio of is_attributed on each day

# In[3]:


df_Test = pd.read_csv(test_fpath, dtype=dtypes, parse_dates=["click_time"])


# In[9]:


for sample_date in ["2017-11-06", "2017-11-07", "2017-11-08", "2017-11-09"]:
    df_Samples = get_samples_by_date(sample_date)
    print(sample_date)
    display(df_Samples.groupby("is_attributed").agg({"ip": ["count", ratio]}))
    gc.collect()


# In[10]:


df_Test.head()


# ## Does training data cover all values in testing data

# In[11]:


testing_data_vals = {}
for col in ["ip", "app", "device", "os", "channel"]:
    testing_data_vals[col] = set(df_Test[col].unique())
for sample_date in tqdm_notebook(["2017-11-06", "2017-11-07", "2017-11-08", "2017-11-09"]):
    df_Samples = get_samples_by_date(sample_date)
    for col in ["ip", "app", "device", "os", "channel"]:
        testing_data_vals[col] = testing_data_vals[col] - set(df_Samples[col].unique())
    gc.collect()


# In[12]:


for col, vals in testing_data_vals.items():
    print("{}: There are {} values don't exist in training data.".format(col, len(vals)))


# In[ ]:




