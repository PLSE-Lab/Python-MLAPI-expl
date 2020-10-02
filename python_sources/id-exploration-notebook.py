#!/usr/bin/env python
# coding: utf-8

# From the data page:
# 
# > "This dataset contains anonymized features pertaining to a time-varying value for a financial instrument. Each instrument has an id."
# 
# This notebook aims to explore the ID component of the data and make some instrument-level plots.
# 
# Forked from [SRK's exploratory notebook][1].
# 
# 
#   [1]: https://www.kaggle.com/sudalairajkumar/two-sigma-financial-modeling/simple-exploration-notebook/notebook

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# Please note that, in this competition HDF5 file is being used instead of csv.

# In[ ]:


with pd.HDFStore("../input/train.h5", "r") as train:
    # Note that the "train" dataframe is the only dataframe in the file
    df = train.get("train")


# In[ ]:


print("Number of instruments:", len(df["id"].unique()))
print("Min ID:", df["id"].min())
print("Max ID:", df["id"].max())


# **There are 1424 unique instrument IDs in the training set.** Given the min and max instrument IDs, we might reasonably expect there to be over 2000 in the train and test sets combined. If this is the case it might be valuable to hold out instruments during validation.
# 
# Does the target vary much by ID? Are some IDs represented more than others?

# In[ ]:


stats = df.groupby("id")["y"].agg({"mean":np.mean, "count":len})
sns.jointplot(x="count", y="mean", data=stats)


# In[ ]:


df.groupby("id")["y"].mean().sort_values().head()


# There are outliers, but since they are are for instruments with very few entries, we may be looking at statistical flukes.
# 
# Let's have a look at a few time series the biggest outlier (ID 1431) and a couple of other instruments.

# In[ ]:


cols_to_use = ['y', 'technical_30', 'technical_20', 'fundamental_11', 'technical_19']
fig = plt.figure(figsize=(8, 20))
plot_count = 0
for col in cols_to_use:
    plot_count += 1
    plt.subplot(5, 2, plot_count)
    plt.plot(df["timestamp"].sample(frac=0.01), df[col].sample(frac=0.01), ".")
    plt.title("Distribution of {}".format(col))
    plot_count += 1
    plt.subplot(5, 2, plot_count)
    plt.plot(df.loc[df["id"]==1431, "timestamp"], df.loc[df["id"]==1431, col], ".-", label="ID 1431")
    plt.plot(df.loc[df["id"]==11, "timestamp"], df.loc[df["id"]==11, col], ".-", label="ID 11", alpha=0.7)
    plt.plot(df.loc[df["id"]==12, "timestamp"], df.loc[df["id"]==12, col], ".-", label="ID 12", alpha=0.7)
    plt.legend()
plt.show()


# There is serious time structure in here! Hopefully this notebook can inspire some feature generation and further exploration. 
# 
# A few questions that come to mind:
# 
#  - Is there any autocorrelation in y when broken down by instrument?
#  - Is the instrument ID alone predictive of y? My guess is probably not.
#  - Will long-term time-structure in the features lead to problems with covariate shift?

# In[ ]:




