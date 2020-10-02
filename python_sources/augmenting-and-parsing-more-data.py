#!/usr/bin/env python
# coding: utf-8

# This kernel is an attempt to list as many as possible ways to augment or parse more data for the "jigsaw II competition".
# Work in progress.

# In[34]:


import numpy as np
import pandas as pd 
import os
import datetime as dt


# ## Data Augmentation

# ### K-nearest neuborghs
# _____________________________
# The idea is, using embedding and k-nn to find "close" words to each other.
# Was tried:
# 
# https://www.kaggle.com/theoviel/using-word-embeddings-for-data-augmentation
# 
# https://www.kaggle.com/shujian/fake-some-positive-data-data-augmentation
# 
# Didn't work for people, who tried, so I've skiped implementation
# 

# ### Marcov chains
# _______________________
# Was tried in
# https://www.kaggle.com/jpmiller/extending-train-data-with-markov-chains
# 
# Didn't work for people, who tried, so I've skiped implementation

# ### Translation
# -------------------------
# Proposed in https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/48038#272289
# by @pavelost
# 
# The code and instruction can be found in my GitHub repo: https://github.com/PavelOstyakov/toxic/tree/master/tools

# # Third party data sources

# ### First jigsaw competition

# In[ ]:


get_ipython().system('ls ../input/jigsaw-toxic-comment-classification-challenge')


# In[ ]:


data = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')
data.head()


# In[ ]:


data['toxic'] = data[data.columns[2:]].sum(axis=1)
toxic = data[data['toxic'] > 0]
not_toxic = data[data['toxic'] == 0]
print(f'toxic comments: {toxic.shape[0]}, normal: {not_toxic.shape[0]}')


# ### Wikipedia Talk Labels: Personal Attacks
# -------------
# https://figshare.com/articles/Wikipedia_Detox_Data/4054689

# In[ ]:


get_ipython().system('ls ../input/wikipedia-talk-labels-personal-attacks/')


# In[ ]:


data_commets = pd.read_csv('../input/wikipedia-talk-labels-personal-attacks/attack_annotated_comments.csv')
data_attack =  pd.read_csv('../input/wikipedia-talk-labels-personal-attacks/attack_annotations.csv')


# In[ ]:


data_attack.drop(columns=data_attack.columns[1:-1], inplace=True)
summery = data_attack.groupby(['rev_id']).sum()
data = data_commets.set_index('rev_id').join(summery)


# In[ ]:


data.head()


# In[ ]:


toxic = data[data['attack'] > 0]
not_toxic = data[data['attack'] == 0]
print(f'toxic comments: {toxic.shape[0]}, normal: {not_toxic.shape[0]}')


# ### Wikipedia Talk Corpus
# -----------
# https://figshare.com/articles/Wikipedia_Talk_Corpus/4264973

# In[ ]:


get_ipython().system('ls ../input/wikipedia-talk-corpus-sample/')


# In[ ]:


# The whole corpus is too large to load in kernel, here is a small sample
data = pd.read_csv('../input/wikipedia-talk-corpus-sample/chunk_0.tsv', sep='\t')
data.head()


# ## Reddit
# --------------------------------
# will use https://github.com/dmarx/psaw 
# A minimalist wrapper for searching public reddit comments/submissions via the pushshift.io API.

# In[ ]:


get_ipython().system('pip install psaw')


# In[ ]:


# Example of general use
from psaw import PushshiftAPI
api = PushshiftAPI()

# The `search_comments` and `search_submissions` methods return generator objects
gen = api.search_submissions(limit=100)
results = list(gen)

# There are 2 main attributes we may be interested in:
# title - provides the title of a submission
print(results[1].title)
# selftext - provides main text, if text exists
print(results[1].selftext)


# ### Toxic Reddit

# [ShitRedditSays](https://www.reddit.com/r/ShitRedditSays/) is
# self appointed reddit hate speech watchdog.
# According to web.archive.org subreddit started in the midst of 2011

# In[ ]:


# Start time
start_epoch=int(dt.datetime(2017, 1, 1).timestamp())
# Found submissions
shit_reddit_says = list(api.search_submissions(after=start_epoch,
                                               subreddit='ShitRedditSays',
                                               filter=['url','author', 'title', 'subreddit'],
                                               limit=10))

# Some questionable comments
for i in range (6):
    print(shit_reddit_says[i].title, '\n')


# ### Wholsome reddit

# #### Male
# * [MensRights](https://www.reddit.com/r/MensRights/)
# * [AskMen](https://www.reddit.com/r/AskMen)
# 
# #### Female
# * [AskWomen](https://www.reddit.com/r/AskWomen)
# * [women](https://www.reddit.com/r/women/)
# * [femenism](https://www.reddit.com/r/femenism/)
# 
# #### homosexual_gay_or_lesbian
# * [lgbt](https://www.reddit.com/r/lgbt/)
# * [gaybros](https://www.reddit.com/r/gaybros/)
# * [LesbianActually](https://www.reddit.com/r/LesbianActually/)
# 
# #### christian
# * [Catholicism](https://www.reddit.com/r/Catholicism/)
# 
# #### Jewish 
# * [Jewish](https://www.reddit.com/r/Jewish/)
# * [Judaism](https://www.reddit.com/r/Judaism/)
# 
# #### muslim
# * [islam](https://www.reddit.com/r/islam/)
# 
# #### black
# * [BlackPeopleTwitter](https://www.reddit.com/r/BlackPeopleTwitter/)
# * [Blackfellas](https://www.reddit.com/r/Blackfellas/)
# * [AsABlackMan](https://www.reddit.com/r/AsABlackMan/)
# 
# #### white
# Wasn't able to find anything particularly positive)))
# 
# #### psychiatric_or_mental_illness
# * [mentalhealth](https://www.reddit.com/r/mentalhealth/)

# In[ ]:


# Start time
start_epoch=int(dt.datetime(2018, 1, 1).timestamp())
# Found submissions
lgbt = list(api.search_submissions(after=start_epoch,
                                               subreddit='lgbt',
                                               filter=['url','author', 'title', 'subreddit'],
                                               limit=10))

# Some positive comments
for i in range (6):
    print(lgbt[i].title, '\n')

