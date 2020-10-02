#!/usr/bin/env python
# coding: utf-8

# ## Simple Exploratory Analysis of tweets

# In this kernel , I have done a exploratory analysis of the real vs non real tweets.I have considered following kernel as references while preparing this kernel.If you consider upvoting my kernel,pls upvote these kernels too.
# 
# 1.https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-qiqc
# 
# 2.https://www.kaggle.com/arthurtok/spooky-nlp-and-topic-modelling-tutorial

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import gc

from wordcloud import STOPWORDS
import string
import seaborn as sns


# In[ ]:


get_ipython().system('ls ../input')


# In[ ]:


DATA_PATH='../input/nlp-getting-started'


# In[ ]:


train=pd.read_csv(f'{DATA_PATH}/train.csv')
test=pd.read_csv(f'{DATA_PATH}/test.csv')


# In[ ]:


train.head()


# In[ ]:


print(f'Train data has {train.shape[0]} rows')
print(f'Test data has {test.shape[0]} rows')


# Let us check the unique keywords found in the train dataset.But before this let us check the number of NA values in that column.

# In[ ]:


train['keyword'].isna().sum()


# In[ ]:


print(f'The percentage of NA values in the keyword column {round((train.keyword.isna().sum()/train.shape[0])*100,2)} %')


# In[ ]:


print(f'There are {train.keyword.nunique()} unique keyword in the column')


# Lets check how many unique keywords are represented for target =1 ie.when the tweet represents real emergency.

# In[ ]:


real_keywords=train.loc[train['target']==1]['keyword'].dropna().unique()
nonreal_keywords=train.loc[train['target']==0]['keyword'].dropna().unique()


# In[ ]:


print(f'There are {len(real_keywords)} real keywords and {len(nonreal_keywords)} non real keywords in the train dataset')


# There is a possibility that the keywords might be the same in case of real as well as non real tweets.Lets check the unique keywords in the real tweets alone.

# In[ ]:


print(f'Unique keywords found in real tweets only {set(real_keywords)-set(nonreal_keywords)}')
print(f'Unique keywords found in non real tweets only {set(nonreal_keywords)-set(real_keywords)}')


# Thus from the above we see that there are 3 keywords found only in the real tweets whereas there is 1 keyword unique to non real tweets .But , this might not provide a holistic view since there are keywords which convey the same meaning , let us take an example of derail.

# In[ ]:


train.loc[train['keyword'].str.startswith('derail',na=False)]['keyword'].unique()


# We see that there are 3 keywords which convey the same meaning of derail out of which the derailment keyword is used in the real tweets only whereas derail and derailed are present in both real and non real tweets.

# In[ ]:


train.loc[train['keyword'].str.startswith('wreck',na=False)]['keyword'].unique()


# Similarly for wreck which is represented in three different ways.Lets check the most used keywords to for real vs non-real emergency.

# In[ ]:


train[train.target==1]['keyword'].dropna().value_counts()[0:5]


# In[ ]:


train[train.target==0]['keyword'].dropna().value_counts()[0:5]


# Now that we have checked on the text columns and create some features which might help during our modelling.I have considered popular kernel of [SRK](https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-qiqc) as a reference for this analysis.The features we will look at are 
# 
# 1.Number of words in the text
# 
# 2.Number of unique words in the text
# 
# 3.Number of characters in the text
# 
# 4.Number of stopwords
# 
# 5.Number of punctuations
# 
# 6.Average length of the words
# 

# In[ ]:


## Creating the features:

train['n_words']=train['text'].apply(lambda x:len(str(x).split()))
train['n_unique_words']=train['text'].apply(lambda x:len(set(str(x).split())))
train['n_characters']=train['text'].apply(lambda x:len(str(x)))
train['n_stopwords']=train['text'].apply(lambda x:len([w for w in str(x).lower().split() if w in STOPWORDS]))
train['n_punctuations']=train['text'].apply(lambda x:len([w for w in str(x) if w in string.punctuation ]))
train['n_avg_words']=train['text'].apply(lambda x:np.mean([len(w) for w in str(x).split()]))


# In[ ]:


train.head()


# In[ ]:


columns=['n_words','n_unique_words','n_characters','n_stopwords','n_punctuations','n_avg_words']


# In[ ]:


for c in columns:
    plt.figure(figsize=(8,8))
    ax=sns.boxplot(x='target',y=c,data=train)
    ax.set_xlabel(xlabel='Target')
    ax.set_ylabel(ylabel=c)
    plt.title(r'Boxplot of {} vs Target'.format(c))
    plt.show()


# **Inference**:
# From the plots it is understood that,
# 
# 1.There is no significant difference between the number of words and the targets.
# 
# 2.The median number of unique words for real tweets is slighly higher than that of non-real tweets.
# 
# 3.The number of characters between real and non-real tweets is significantly different.The median number of characters for real tweets is higher.
# 
# 4.There are many outliers when it comes to the number of stopwords and non-real tweets seems to have more of them when compared to real tweets.
# 
# 5.Non-real tweets sentences have lot of outliers in the number of punctuations( going upto 60) when compared to real tweets.
# 
# 6.The median value of real tweets for number of average words is higher compared to non real tweets.This distribution is also with lot of outliers.

# **work in progress**
