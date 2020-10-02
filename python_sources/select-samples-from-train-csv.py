#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import defaultdict
fpath = "../input/train.csv"


# ## How many samples in this file

# In[2]:


with open(fpath) as f:
    columns = next(f).strip().split(",")
    sample_counts = 0
    click_dates = set([])
    for line in f:
        sample_counts += 1
        click_dates.add(line.split(",")[-3][:10])
print("There are {} samples in this file including {} unique dates.".format(sample_counts, len(click_dates)))


# ## How many samples on each day

# In[4]:


with open(fpath) as f:
    _ = next(f)
    sample_counts_on_each_day = defaultdict(int)
    for line in f:
        click_date = line.split(",")[-3][:10]
        sample_counts_on_each_day[click_date] += 1
for date, count in sample_counts_on_each_day.items():
    print("{}: {}".format(date, count))


# ## Get samples on a specific date

# In[5]:


def get_samples_by_date(at):
    nrows = 0
    skiprows = 1
    for date, count in sample_counts_on_each_day.items():
        if date == at:
            nrows = count
        elif date < at:
            skiprows += count
    return pd.read_csv(fpath, skiprows=skiprows, nrows=nrows, names=columns, parse_dates=["click_time"])
df_Samples = get_samples_by_date("2017-11-06")


# In[ ]:


df_Samples.shape


# In[6]:


df_Samples.head()


# In[7]:


df_Samples.tail()


# In[ ]:




