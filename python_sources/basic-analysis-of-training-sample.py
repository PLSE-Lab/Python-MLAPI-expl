#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# # Get a sense from the training samples
# ---
# 1. The target variable is extremely imbalanced
# 2. The most downloaded app often have very low probability to be downloaded, lower than 0.1%.
# 3. By looking at precision of recall of each single feature, we may use the product of recall and precision as the encoding method to represent each categorical value.

# In[3]:


df_train_sample = pd.read_csv("../input/train_sample.csv")


# In[4]:


print("Training Samples contains {} rows {} columns".format(*df_train_sample.shape))


# In[5]:


df_train_sample.head()


# In[6]:


# It's extremely imbalanced
df_train_sample.groupby("is_attributed").agg({"ip": "count"})


# In[7]:


def get_precision(x):
    return x.sum() / x.shape[0]
def get_precision_recall_by_single_feature(col, df):
    df_precision_recall = pd.DataFrame(columns=["Count", "Precision", "Recall"], index=df[col].unique())
    for c, f in [("Precision", get_precision), ("Count", "count"), ("Recall", "sum")]:
        _df = df.groupby(col).agg({"is_attributed": f})
        df_precision_recall.loc[_df.index, c] = _df["is_attributed"]
    df_precision_recall["Recall"] = df_precision_recall["Recall"] / df["is_attributed"].sum()
    return df_precision_recall


# ## Is APP a good feature

# In[8]:


col = "app"
df_app_precision_recall = get_precision_recall_by_single_feature(col, df_train_sample)
# Get top 5 most indicative app
display(df_app_precision_recall.sort_values("Precision", ascending=False).head())
# Get top 5 most download app
display(df_app_precision_recall.sort_values("Recall", ascending=False).head())


# ## Is DEVICE a good feature

# In[9]:


col = "device"
df_app_precision_recall = get_precision_recall_by_single_feature(col, df_train_sample)
# Get top 5 most indicative app
display(df_app_precision_recall.sort_values("Precision", ascending=False).head())
# Get top 5 most download app
display(df_app_precision_recall.sort_values("Recall", ascending=False).head())


# ## Is OS a good feature

# In[10]:


col = "os"
df_app_precision_recall = get_precision_recall_by_single_feature(col, df_train_sample)
# Get top 5 most indicative app
display(df_app_precision_recall.sort_values("Precision", ascending=False).head())
# Get top 5 most download app
display(df_app_precision_recall.sort_values("Recall", ascending=False).head())


# ## Is CHANNEL a good feature

# In[11]:


col = "channel"
df_app_precision_recall = get_precision_recall_by_single_feature(col, df_train_sample)
# Get top 5 most indicative app
display(df_app_precision_recall.sort_values("Precision", ascending=False).head())
# Get top 5 most download app
display(df_app_precision_recall.sort_values("Recall", ascending=False).head())


# In[ ]:




