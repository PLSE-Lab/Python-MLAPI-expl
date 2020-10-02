#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df_train = pd.read_csv('../input/train.csv',index_col=0)


# ## Class Imbalance
# 
# Class imbalance is something to be aware of when training.

# In[ ]:


sns.countplot(x='TARGET',data=df_train,palette="husl", order = range(2))


# ## Missing Data
# 
# First, let's find if any column with missing data.

# In[ ]:


# Let's look at the size of the train dataset
print("train:  nrows %d, ncols %d" % df_train.shape)
# See nan proportion per columns
print('# of nan values in train set : ')
print(df_train.isnull().sum(axis = 0).sort_values(ascending = False).head(10))


# ## Constant columns
# We should remove constant columns

# In[ ]:


# remove constant columns
remove = []
for col in df_train.columns:
    if df_train[col].std() == 0:
        remove.append(col)
print("Constant columes: ",remove)
df_train.drop(remove, axis=1, inplace=True)


# ## Almost all zero columns
# Let us have look at almost all zeron columns

# In[ ]:


# ratio of nonzero elements
plt.rcParams['figure.figsize'] = (14.0, 10.0)
num_non_zero=np.sum(df_train!=0,axis=0).sort_values(ascending = True)
num_non_zero.plot()


# In[ ]:


num_non_zero_col = list(num_non_zero.index[num_non_zero<=19])
print("Almost all zeros columes: ",num_non_zero_col)
not_zero_rows = np.logical_and(np.any(df_train[num_non_zero_col]!=0,axis=1),df_train["TARGET"])
num_non_zero_col.append("TARGET")
almost_zeros = df_train.loc[not_zero_rows,num_non_zero_col]
print(almost_zeros.shape)
num_non_zero_col.remove("TARGET")
df_train.drop(num_non_zero_col, axis=1, inplace=True)


# ## Duplicated columns
# We should remove all duplicated columns

# In[ ]:


# remove duplicated columns
remove = []
c = df_train.columns
for i in range(len(c)-1):
    v = df_train[c[i]].values
    for j in range(i+1,len(c)):
        if np.array_equal(v,df_train[c[j]].values):
            remove.append(c[j])
print("Duplicated columes: ",remove)
df_train.drop(remove, axis=1, inplace=True)
# Let's look at the size of the train dataset
print("After simple preprocess, train:  nrows %d, ncols %d" % df_train.shape)


# ## Plot Heatmap of correlation matrix
# Since there are too many features, we split the columns into 25 sub-figures.

# In[ ]:


cor_mat = df_train.corr()
for i in range(5):
    for j in range(5):
        x = i*50
        y = j*50
        corr = cor_mat.iloc[range(x,x+50),range(y,y+50)]
        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(15, 12))
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr,linewidths=.5, ax=ax)

