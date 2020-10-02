#!/usr/bin/env python
# coding: utf-8

# ##Attention
# 
# Thanks to comments on this kernel, I found some problem.
# 
# *In this kernel, the datetime of transaction is not seriously discussed.
# 
# *Transactions which have the identity data are more doubtful.
# 
# In V3, I fixed about second problem, but still have the first problem
# 
# And that, I mainly focused on getting interesting information, not on getting useful information only for this competition. 

# **General**
# 
# After provoking the discussion (https://www.kaggle.com/c/ieee-fraud-detection/discussion/103565), I want to examine it.
# 
# I wish that this kernel would help your research! 
# 

# I focused on browser!
# 
# I am a very lazy person, so I rarely update my browser.
# 
# How swindlers are? 

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


# In[ ]:





# In[ ]:


df.head()


# In[ ]:


pd.DataFrame(df.groupby("id_31").count()["TransactionID"]).head()


# There are a lot of data from which we cannot say that someone are try hard to update browser.
# 
# Therefore, most of them was NaN.

# In[ ]:


a = np.zeros(10)
a[:] = np.nan##this is numpy structure


# In[ ]:


a


# In[ ]:


a = np.zeros(df.shape[0])
a = np.nan
df["lastest_browser"] = a


# In[ ]:


df.head()


# In[ ]:


df.lastest_browser[df["id_31"]=="samsung browser 7.0"]=1
df.lastest_browser[df["id_31"]=="opera 53.0"]=1
df.lastest_browser[df["id_31"]=="mobile safari 10.0"]=1
df.lastest_browser[df["id_31"]=="google search application 49.0"]=1
df.lastest_browser[df["id_31"]=="firefox 60.0"]=1
df.lastest_browser[df["id_31"]=="edge 17.0"]=1
df.lastest_browser[df["id_31"]=="chrome 69.0"]=1
df.lastest_browser[df["id_31"]=="chrome 67.0 for android"]=1
df.lastest_browser[df["id_31"]=="chrome 63.0"]=1
df.lastest_browser[df["id_31"]=="chrome 63.0 for android"]=1
df.lastest_browser[df["id_31"]=="chrome 63.0 for ios"]=1
df.lastest_browser[df["id_31"]=="chrome 64.0"]=1
df.lastest_browser[df["id_31"]=="chrome 64.0 for android"]=1
df.lastest_browser[df["id_31"]=="chrome 64.0 for ios"]=1
df.lastest_browser[df["id_31"]=="chrome 65.0"]=1
df.lastest_browser[df["id_31"]=="chrome 65.0 for android"]=1
df.lastest_browser[df["id_31"]=="chrome 65.0 for ios"]=1
df.lastest_browser[df["id_31"]=="chrome 66.0"]=1
df.lastest_browser[df["id_31"]=="chrome 66.0 for android"]=1
df.lastest_browser[df["id_31"]=="chrome 66.0 for ios"]=1


# I consulted with wikipedia about chrome.
# 
# https://en.wikipedia.org/wiki/Google_Chrome_version_history
# On December 5th in 2017, chrome 63 version was released.
# 
# I determined 63 was lastest.
# 
# It may be more useful if you use the date information.
# 

# In[ ]:


df.lastest_browser[df.id_31=="samsung browser 7.0"]
df.lastest_browser[df["id_31"]=="chrome 66.0 for ios"]


# **Let's check!**

# In[ ]:


df.isFraud.mean()


# In[ ]:


df.groupby("lastest_browser").mean()["isFraud"]


# In[ ]:


df["null"]=df.id_01.isnull()


# In[ ]:


df.groupby("null").mean()["isFraud"]


# ##Conclusion

# This result suggests that people who update browser are more possibly fraud makers than people who are lazy about browser. 
# 
# Fraud makers may be earnest people.
# 

# In[ ]:




