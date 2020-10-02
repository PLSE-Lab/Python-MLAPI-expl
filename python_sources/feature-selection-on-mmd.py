#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectPercentile, chi2
import time


# In[2]:


asm_final_features = pd.read_csv('../input/asm_final_features.csv', index_col = 0 )


# In[3]:


asm_final_features = asm_final_features.set_index("Id")


# In[4]:


asm_final_features.head()


# In[5]:


asm_final_features = asm_final_features.dropna(axis=1)


# In[6]:


asm_final_features.shape


# # Chi-Square for Feature Reduction on ASM Files

# In[7]:


asm_final_features = asm_final_features.reset_index()


# In[8]:


asm_final_features.shape


# In[9]:


labels = pd.read_csv("../input/trainLabels.csv")


# In[10]:


asm_final_features = pd.merge(asm_final_features, labels, on = "Id")


# In[11]:


y = asm_final_features["Class"]


# In[12]:


X = asm_final_features.drop(["Id", "Class"], axis = 1)


# In[13]:


model = SelectPercentile(chi2, percentile = 50)
X_new = model.fit_transform(X,y)


# In[14]:


X_new.shape


# In[15]:


reduced_df = X.iloc[:, model.get_support(indices=True)]


# In[16]:


useful_features = list(reduced_df.columns)
useful_features.insert(0, "Id")


# In[18]:


asm_reduced_final = asm_final_features[useful_features]


# In[19]:


asm_reduced_final = asm_reduced_final.set_index("Id")


# In[20]:


asm_reduced_final.to_csv("./asm_reduced_final.csv")


# In[21]:


asm_reduced_final = pd.read_csv("./asm_reduced_final.csv")


# In[23]:


asm_reduced_final.shape


# **'asm_reduced_final.csv'** consists of final reduced features of asm files.

# In[ ]:




