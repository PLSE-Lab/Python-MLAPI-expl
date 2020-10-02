#!/usr/bin/env python
# coding: utf-8

# The main  objective of the  competetion is to find whether the trasaction of the customer is potential or not. So first lets explore the data.

# In[41]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


# In[42]:


from subprocess import check_output
print(check_output(["ls", "../input/"]).decode("utf8"))


# In[43]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[44]:


print("Train rows and columns : ", train.shape)
print("Test rows and columns : ", test.shape)


# Here number of samples are less and number of features are high in train

# In[45]:


train.head()


# In[46]:


train.dtypes


# In[47]:


train.info()


# In[48]:


train.describe()


# It is quite strange that we dont have any column names here, so what does they mean actually

# In[49]:


plt.hist(train.target.values)


# In[50]:


train.target.sort_values(ascending=False)


# In[51]:


train.target.value_counts().sort_values(ascending=False)


# In[52]:


sns.distplot(train.target)
plt.title('Target histogram.');


# In[53]:


train.target = np.log10(train.target)


# In[54]:


sns.distplot(train.target)
plt.title('Logarithm transformed target histogram.');


# In[55]:


plt.figure(figsize=(8,6))
plt.scatter(range(train.shape[0]), np.sort(train['target'].values))
plt.xlabel('index', fontsize=12)
plt.ylabel('Target', fontsize=12)
plt.title("Target Distribution", fontsize=14)
plt.show()


# In[56]:


unique_df = train.nunique().reset_index()
unique_df


# In[57]:


unique_df.columns = ["col_name", "unique_count"]
constant_df = unique_df[unique_df["unique_count"]==1]
constant_df.shape


# more in pipeline, if you like it please upvote for me .
# 
# Thank you : )
