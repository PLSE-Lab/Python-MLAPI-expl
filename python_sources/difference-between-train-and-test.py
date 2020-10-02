#!/usr/bin/env python
# coding: utf-8

# plot feature difference between train and test

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


dtype = {
    "ip": "uint32",
    "app": "uint16",
    "device": "uint16",
    "os": "uint16",
    "channel": "uint16",
}
train = pd.read_csv("../input/train.csv", dtype=dtype)
test = pd.read_csv("../input/test.csv", dtype=dtype)


# ## ip

# In[3]:


fig, (axl, axr) = plt.subplots(ncols=2, figsize=(15,6))
axl.set_title("train")
axr.set_title("test")
sns.distplot(train["ip"], kde=False, ax=axl)
sns.distplot(test["ip"], kde=False, ax=axr)
plt.plot()


# In[4]:


df = pd.concat([train["ip"].describe(), test["ip"].describe()], axis=1)
df.columns = ["train", "test"]
df


# In[5]:


# train only
len(set(train["ip"]) - set(test["ip"]))


# In[6]:


# test only
len(set(test["ip"]) - set(train["ip"]))


# ## app

# In[7]:


fig, (axl, axr) = plt.subplots(ncols=2, figsize=(15,6))
axl.set_title("train")
axr.set_title("test")
sns.distplot(train["app"], kde=False, ax=axl)
sns.distplot(test["app"], kde=False, ax=axr)
plt.plot()


# In[8]:


df = pd.concat([train["app"].describe(), test["app"].describe()], axis=1)
df.columns = ["train", "test"]
df


# In[9]:


# train only
len(set(train["app"]) - set(test["app"]))


# In[10]:


# test only
len(set(test["app"]) - set(train["app"]))


# ## device

# In[11]:


fig, (axl, axr) = plt.subplots(ncols=2, figsize=(15,6))
axl.set_title("train")
axr.set_title("test")
sns.distplot(train["device"], kde=False, ax=axl)
sns.distplot(test["device"], kde=False, ax=axr)
plt.plot()


# In[12]:


df = pd.concat([train["device"].describe(), test["device"].describe()], axis=1)
df.columns = ["train", "test"]
df


# In[13]:


# train only
len(set(train["device"]) - set(test["device"]))


# In[14]:


# test only
len(set(test["device"]) - set(train["device"]))


# ## os

# In[15]:


fig, (axl, axr) = plt.subplots(ncols=2, figsize=(15,6))
axl.set_title("train")
axr.set_title("test")
sns.distplot(train["os"], kde=False, ax=axl)
sns.distplot(test["os"], kde=False, ax=axr)
plt.plot()


# In[16]:


df = pd.concat([train["os"].describe(), test["os"].describe()], axis=1)
df.columns = ["train", "test"]
df


# In[17]:


# train only
len(set(train["os"]) - set(test["os"]))


# In[18]:


# test only
len(set(test["os"]) - set(train["os"]))


# ## channel

# In[19]:


fig, (axl, axr) = plt.subplots(ncols=2, figsize=(15,6))
axl.set_title("train")
axr.set_title("test")
sns.distplot(train["channel"], kde=False, ax=axl)
sns.distplot(test["channel"], kde=False, ax=axr)
plt.plot()


# In[20]:


df = pd.concat([train["channel"].describe(), test["channel"].describe()], axis=1)
df.columns = ["train", "test"]
df


# In[21]:


# train only
len(set(train["channel"]) - set(test["channel"]))


# In[22]:


# test only
len(set(test["channel"]) - set(train["channel"]))


# ## click time

# In[23]:


# train
pd.to_datetime(train["click_time"]).describe()


# In[24]:


# test
pd.to_datetime(test["click_time"]).describe()

