#!/usr/bin/env python
# coding: utf-8

# The idea is **visualizing** features with similar basic stats **patterns**. 
# Maybe some of them correspond to the same variable in different time steps?
# 
# Do you think I should standardize the variables because they could have been transformed to different ranges?
# 

# In[13]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


# In[14]:


#Load the Data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
features = [c for c in train.columns if c not in ['ID_code', 'target']]


# In[15]:


res=train.describe()
res


# Let's apply **hierarchical clustering** with eucliden distance and average method:

# In[19]:


g = sns.clustermap(res.iloc[1:,:], figsize=(30,15));


# Mmm.. interesting. Why some of them are so similar?
# Let's explore if this can help for feature engineering. I will consider one of the clusters of features with similar stats as an example of the possible features that can be generated: 
# ['var_86','var_76','var_139','var_80','var_123']

# In[17]:


col=['var_86','var_76','var_139','var_80','var_123']


# In[18]:


#Considering the mean per row of all variables, the distributions are pretty similar:
sns.distplot(train.loc[train['target']==0,:].mean(axis=1),bins=15)
sns.distplot(train.loc[train['target']==1,:].mean(axis=1),bins=15);


# In[ ]:


#But, if we calculate the mean of these variables we start to see some difference:
sns.distplot(train.loc[train['target']==0,col].mean(axis=1),bins=15)
sns.distplot(train.loc[train['target']==1,col].mean(axis=1),bins=15);


# In[ ]:


sns.distplot(train.loc[train['target']==0,col].min(axis=1),bins=15)
sns.distplot(train.loc[train['target']==1,col].min(axis=1),bins=15);


# Please, if this helps you to get any idea, share it with us! ;)
