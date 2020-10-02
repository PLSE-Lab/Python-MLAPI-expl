#!/usr/bin/env python
# coding: utf-8

# # Google QUEST Q&A Labeling
# This notebook performs the micro EDA of the Contest dataset of Google QUEST Q&A Labeling
# In this notebook I have tried to perform a extremely basic Data Analysis of the provided data. If you have suggestion or willing to correct me anywhere in the comments.
# 

# In[ ]:


from pathlib import Path
import os
import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


RANDOM_SEED = 123
BASE=Path('../input/google-quest-challenge')
for i in os.walk(os.path.join(BASE)):
    print(i)


# In[ ]:


train_df=pd.read_csv(BASE/'train.csv')
train_df.tail()


# In[ ]:


test_df=pd.read_csv(BASE/'test.csv')
test_df.tail()


# In[ ]:


# Viewing the value of first row of the training data
for i,j in train_df.iloc[0].items():
#     print(i.ljust(30),j)
    print('-'*80)
    print(i)
    print('-'*80)
    print(j,'\n')


# In[ ]:


# Plotting the channels where the Training queries comes from in the data
fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111)
width = 0.4
train_df.host.value_counts().plot(kind='bar', color='blue', ax=ax, width=width, position=1)
test_df.host.value_counts().plot(kind='bar', color='red', ax=ax, width=width, position=0)
ax.set_xlabel('Sites')
ax.set_ylabel('Question Counts')


# In[ ]:


# Plotting the channels where the testing queries comes from in the data
plt.figure(figsize=(16,5))
width = 0.4
test_df.host.value_counts().plot(kind='bar', color='red', width=width)


# In[ ]:


# Plotting the category occurance of the queries
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
width = 0.2
train_df.category.value_counts().plot(kind='bar', color='blue', ax=ax, width=width, position=1, legend=True)
test_df.category.value_counts().plot(kind='bar', color='red', ax=ax, width=width, position=0, legend=True)
ax.set_xlabel('Sites')
ax.set_ylabel('Question Counts')


# In[ ]:


# Finding the size of longest query in the table
sentence_len = train_df.answer.apply(lambda x: len(x))
sentence_len.max()


# In[ ]:


# Prints column with longest query
train_df[sentence_len==22636]


# I hope you find it useful. Please drop your comment and help me improve in my future kernels. Also if you find it informative, do **UPVOTE**.
