#!/usr/bin/env python
# coding: utf-8

# **Load packages**

# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")

# Print all rows and columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# **Read the data**

# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


# In[ ]:


print("Dimension of train :",train_df.shape)
print("Dimension of test :",test_df.shape)


# **Observations**: 
# 1. The column number exceeds the rows number for the train data. 
# 2. The test data is containing almost 10 times more data than the train data.

# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


train_df.info()


# In[ ]:


test_df.info()


# **Check for missing value**

# In[ ]:


train_df.isnull().values.any()


# In[ ]:


test_df.isnull().values.any()


# **Check for Sparsity**

# In[ ]:


def check_sparsity(df):
    non_zeros = (df.ne(0).sum(axis=1)).sum()
    total = df.shape[1]*df.shape[0]
    zeros = total - non_zeros
    sparsity = round(zeros / total * 100,2)
    density = round(non_zeros / total * 100,2)

    print(" Total:",total,"\n Zeros:", zeros, "\n Sparsity [%]: ", sparsity, "\n Density [%]: ", density)

check_sparsity(train_df)


# In[ ]:


check_sparsity(test_df)


# Observation: test data has more sparsity compared to train data

# **Check for Target Variable**

# In[ ]:


train_df['target'].describe()


# In[ ]:


#Distribution plot of target variable
plt.figure(figsize=(8,5))
sns.distplot(train_df['target'])


# This seems to be a highly skewed target variable. Let's take the log of it to check the distribution

# In[ ]:


plt.figure(figsize=(8,5))
sns.distplot(np.log(train_df['target']), kde='False')


# Now distribution of target variable looks much better

# **Prepare the data**

# In[ ]:


X_train = train_df.drop(["ID", "target"], axis=1)
y_train = np.log1p(train_df["target"].values)

X_test = test_df.drop(["ID"], axis=1)


# In[ ]:


X_train.shape,X_test.shape


# **Drop constant feature**

# In[ ]:


drop_cols=[]
for cols in X_train.columns:
    if X_train[cols].std()==0:
        drop_cols.append(cols)
print("Number of constant columns dropped: ", len(drop_cols))
print(drop_cols)


# In[ ]:


X_train.drop(drop_cols,axis=1, inplace = True)
X_test.drop(drop_cols, axis=1, inplace = True)


# In[ ]:


X_train.shape,X_test.shape


# In[ ]:




