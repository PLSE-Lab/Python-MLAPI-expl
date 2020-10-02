#!/usr/bin/env python
# coding: utf-8

# # Introduction
# A lot of people found that there is a lot of similar columns in this dataset.
# 
# https://www.kaggle.com/c/ieee-fraud-detection/discussion/100197#latest-593070
# 
# In this disucussion, that is discussed. 
# 
# V319-V321 is also good example of them.
# 
# How should we "use" it?
# 
# Most people throw away the data, but I think that I can get useful information from it.
# 
# 
# 
# *Attention
# 
# *I'm a data science "novice" which stated a few days ago.
# 
# *I wish that you kindly check and say to fix my kernel.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_rows', 500)
pd.get_option("display.max_columns",500)


# In[ ]:


folder_path = '../input/'
train_identity = pd.read_csv(f'{folder_path}train_identity.csv')
train_transaction = pd.read_csv(f'{folder_path}train_transaction.csv')


df = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')


# # Look the data

# In[ ]:


df.corr().head()


# In[ ]:


cor = df.corr()


# In[ ]:


cor_use = cor[cor>0.8]


# In[ ]:


for i in range(cor_use.shape[0]):
    cor_use.iloc[i,i] = np.nan


# In[ ]:


cor_use.head()


# In[ ]:


corr_use = cor_use[cor_use>0.7].dropna(how='all').dropna(how='all',axis=1)


# In[ ]:


corr_use.shape


# Threre a lot of similar columns.

# In[ ]:


df.loc[:,["V319","V320","V321"]].corr()


# We can find that "V319"-"V321" are also very similar.
# 
# This time, I want to check them.

# # V319-321

# In[ ]:


df.loc[:,['V319','V320','V321']].head(30)


# **They are very similar to each other.**

# In[ ]:


df["diff_V319_V320"] = np.zeros(df.shape[0])
df["diff_V320_V321"] = np.zeros(df.shape[0])
df["diff_V319_V321"] = np.zeros(df.shape[0])


# # V319 - V320

# In[ ]:


len(df[(df["V319"]!=df["V320"])])/df.shape[0]


# We can find that part of data have different values.

# In[ ]:


df.loc[df["V319"]!=df["V320"],"diff_V319_V320"] = 1


# In[ ]:


df[(df["V319"]!=df["V320"])].head()


# In[ ]:


df.groupby("diff_V319_V320").mean().isFraud


# In[ ]:


df.groupby("diff_V319_V320").mean().isFraud.plot()


# We find that transaction which has different V319-V320 is doubtful.
# 
# If you think this column is useful, please use it instead of delete the V319,V320 columns.

# # V320 - V321

# In[ ]:


len(df[(df["V320"]!=df["V321"])])/df.shape[0]


# In[ ]:


df.loc[df["V321"]!=df["V320"],"diff_V320_V321"] = 1


# In[ ]:


df[(df["V321"]!=df["V320"])].head()


# In[ ]:


df.groupby("diff_V320_V321").mean().isFraud


# The similar result is gotten.
# 
# However, the gap is smaller.

# # V319 - V321

# In[ ]:


len(df[(df["V319"]!=df["V321"])])/df.shape[0]


# In[ ]:


df.loc[df["V321"]!=df["V319"],"diff_V319_V321"] = 1


# In[ ]:


df[(df["V321"]!=df["V320"])].head()


# In[ ]:


df.groupby("diff_V320_V321").mean().isFraud


# # Conclusion

# Some people think the gap is little, others think not.
# 
# If you interested in it, please use for your model and judge wheter these columns are useful for your model.
