#!/usr/bin/env python
# coding: utf-8

# In[81]:


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


# In[82]:


df_train = pd.read_csv("../input/train.csv") 


# In[83]:


df_train.shape


# In[84]:


df_train.head()


# **Molecule Name column exploration**

# In[85]:


molecule_name_counts = df_train["molecule_name"].value_counts()
molecule_name_counts


# In[86]:


print("Max: " + molecule_name_counts.idxmax() + ": " + str(molecule_name_counts[molecule_name_counts.idxmax]))
print("Min: " + molecule_name_counts.idxmin() + ": " + str(molecule_name_counts[molecule_name_counts.idxmin]))


# In[87]:


molecule_str = pd.DataFrame(molecule_name_counts.keys().str.split("_").tolist(), columns=["str", "digit"])


# In[88]:


molecule_str.head()


# In[89]:


molecule_str["str"].value_counts()


# Seems like the string is actually the same every where, so we can easily consider it as useless for now. Of course, we should check if this is the case in the test set as well. 
# Let us append the "digit" column to the main test_df

# In[90]:


molecule_str_train = pd.DataFrame(df_train["molecule_name"].str.split("_").tolist(), columns=["str", "digit"])
df_train["molecule_digit"] = molecule_str_train["digit"]


# In[91]:


df_train.head()


# **Scalar coupling constant**

# In[92]:


print("Max: " + str(df_train["scalar_coupling_constant"].loc[df_train["scalar_coupling_constant"].idxmax()]))
print("Min: " + str(df_train["scalar_coupling_constant"].loc[df_train["scalar_coupling_constant"].idxmin()]))


# In[93]:


plt.figure(figsize=(15,20))
df_train["scalar_coupling_constant"].plot.hist(bins=range(-50, 240, 10))


# Most values are around 0. It is interesting how there is almost no value between 20 & 70, then grouped around 70. Then no values between 150 & 190. Let us check this. 

# In[94]:


# Check values between 20 & 70. 
plt.figure(figsize=(15,20))
df_train.loc[ (df_train["scalar_coupling_constant"] > 20) & (df_train["scalar_coupling_constant"] < 70)]["scalar_coupling_constant"].plot.hist(bins=range(20, 70, 10))


# Counts are much less than the global hist, but values are still represented. 

# In[95]:


# Check values between 20 & 70. 
plt.figure(figsize=(15,20))
df_train.loc[ (df_train["scalar_coupling_constant"] > 150) & (df_train["scalar_coupling_constant"] < 190)]["scalar_coupling_constant"].plot.hist(bins=range(150, 190, 10))


# The difference is much more visible here. Only 200 vals between 150 & 160, & less than 20 for higher vals. 

# **Type column exploration**

# In[96]:


from matplotlib import pyplot as plt
plt.figure(figsize=(15,20))
df_train["type"].value_counts().plot.bar()


# **Structure File**

# In[99]:


struct_df = pd.read_csv("../input/structures.csv")


# In[100]:


struct_df.shape


# In[101]:


display(struct_df.head())


# To be continued...
