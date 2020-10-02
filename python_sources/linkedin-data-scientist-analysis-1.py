#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


from subprocess import check_output
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
print(check_output(["ls","../input"]).decode("utf-8"))


# In[3]:


data = pd.read_csv("../input/multipleChoiceResponses.csv",encoding="ISO-8859-1")


# In[4]:


data.head()


# In[96]:


print(data.shape)
print(data.columns)
plt.style.use('fivethirtyeight')


# In[97]:


plt.figure(figsize=(20,12))
df1=data.groupby(["Country"])["Age"].median().sort_values(ascending=False).reset_index()
plt.barh(df1["Country"],df1["Age"],color="rgby")


# In[ ]:





# In[103]:


data["GenderSelect"].value_counts().reset_index()
#plt.batplot(df2["Index"])


# In[104]:


plt.figure(figsize=(12,12))
data.groupby(["Country"])["GenderSelect"].value_counts()


# In[105]:


df3=data.groupby(data["GenderSelect"])["Age"].mean().reset_index()
plt.barh(df3["GenderSelect"],df3["Age"],color="ygbr")


# **Top 15 countries with Highest Respondent**

# In[118]:


plt.figure(figsize=(16,3))
df4= data["Country"].value_counts().reset_index()[:15]
sns.barplot(df4["index"] , df4["Country"])
plt.xticks(rotation=90)


# In[141]:


data["CompensationAmount"]=data["CompensationAmount"].str.replace(",","")
data["CompensationAmount"]=data["CompensationAmount"].str.replace("-","")
#data.groupby(data["Country"])["CompensationAmount"]


# In[ ]:





# In[ ]:




