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


# So here is a kernel for an idea I had a while ago but am still figuring out the finer details for. If you have any tips then I'd love to hear them in the comments. I will also endeavor to add some more text in here
# 
# You can think of it as a probablistic search over a weighted trie (I'll explain what I mean by this in a minute).
# 
# The geneal idea here is to use a [trie](https://en.wikipedia.org/wiki/Trie) to store some corpus of words. During the loading (/learning/whatever) phase we're going to store the occurance count for each letter.
# 
# When we come to try and autocomplete a word we're going to:
# 
# 1. go through the trie and try to find where the end of the input lives in our structure (if the given word doesn't exist we return an error)
# 2. do a breadth first search over the next set of possible characters and keep only the characters most likely to appear next (this is what you need the occurance count for)
# 3. put any completed words (signified by `<eow>` below) in a list and keep note of it's score
# 
# A couple of nice features of this way of doing autocomplete are
# 
# - you can online train the trie. If a user enters a word you've not seen before you could update the trie there and then
# - it does a very limited search and tries to only pick the branches which look to have the highest chance of returning something relevent
# 
# TODOs and things I'd like to investgate
# 
# - different ways of scoring and deciding what to keep in the output list. It's possible to ignore a high scoring word simply because the list of words to return is fulfilled
# - I think maybe a slightly different representation of the Trie would allow picking the next subset of words to look at quicker
# - generally clean up the code a bit - the search function is a bit of a brain dump
# - there might be a nice (although constant) speed up from using numpy in some of the datastructures used here

# In[28]:


import string
import pandas as pd

df_IV = pd.read_table("../input/SW_EpisodeIV.txt", error_bad_lines=False)
df_V = pd.read_table("../input/SW_EpisodeV.txt", error_bad_lines=False)
df_VI = pd.read_table("../input/SW_EpisodeVI.txt", error_bad_lines=False)


# In[29]:


pd.set_option('display.max_colwidth', -1)
df_IV.columns = ['text']
df_V.columns = ['text']
df_VI.columns = ['text']

df_IV.head(5)


# In[30]:


def prep_text(in_text):
    return in_text.split('"')[3:-1][0].lower().translate(str.maketrans("", "", string.punctuation)).split()


# First let's clean the text a bit

# In[31]:


df_IV['clean_text'] = df_IV.apply(lambda row: prep_text(row['text']), axis=1)
df_V['clean_text'] = df_V.apply(lambda row: prep_text(row['text']), axis=1)
df_VI['clean_text'] = df_VI.apply(lambda row: prep_text(row['text']), axis=1)
df_IV.head(5)


# In[32]:


df = pd.concat([df_IV, df_V, df_VI])


# In[33]:


sentences = list()

for idx, row in df.iterrows():
    sentences.append(row['clean_text'])


# In[34]:


sentences[:3]


# In[35]:


flat_list = [item for sublist in sentences for item in sublist]


# In[36]:


df_clean = pd.DataFrame(flat_list)
df_clean.columns = ['clean']


# In[37]:


df_clean['lengths'] = df_clean.apply(lambda row: len(row['clean']), axis=1)


# So what does the distribution of word lengths look like? As you can see it's pretty heavily weighted to words of less than about 5 characters

# In[38]:


df_clean['lengths'].plot(kind='hist')


# In[39]:


y = []

for item in df_clean['clean'].values:
    l = list(item)
    l.append("<eow>")
    y.append(l)


# In[40]:


y[:5]


# In[41]:


out_len = len(df_clean['clean'].unique())
out_len


# In[42]:


class WeightedTrie:
    def __init__(self):
        self.count = 1
        self.tails = {}


# In[43]:


def create_trie(in_text):
    trie = WeightedTrie()

    for word in y:
        curr = trie
        for letter in word:
            if letter in curr.tails:
                curr = curr.tails[letter]
                curr.count += 1
            else:
                new_trie = WeightedTrie()
                curr.tails[letter] = new_trie
                curr = new_trie
                
    return trie


# In[44]:


trie = create_trie(y)


# In[55]:


def predict(word_start, dic, n=3, width=26):
    cs = list(word_start)
    curr = dic
    # start working your way through the trie
    for c in cs:
        if c in curr.tails:
            curr = curr.tails[c]
        else:
            return "word not found, perhaps add it to known words?"
            
    # so at this point we're part way though the trie. In a lot of cases we haven't
    # finished fining the word we want - so let's do a BFS over the nodes at the next
    # layer, but only keep the most likely words to appear next (ie the words with
    # the highest count
    
    # basically BFS
    topn = []
    scores = []
    node_queue = []
    tmp = []
    
    for k, v in curr.tails.items():
        # there are a few different ways to score this. here we penalise longer words
        score = v.count / len(cs + [k])
        tmp.append((score, cs + [k], v.tails))
            
    # todo have a think about this - there might be a nicer way to do this
    # this takes O(W * N lg N) 
    tmp2 = sorted(tmp, reverse=True)
    
    for item in tmp2[:width]:
        node_queue.append(item)
    
    while node_queue and len(topn) < n:
        current_score, so_far, next_dic = node_queue.pop(0)

        if so_far[-1] == '<eow>':
            topn.append(so_far)
            scores.append(current_score)
    
        tmp = []
        for k, v in next_dic.items():
            score = (current_score + v.count) / len(so_far + [k])
            tmp.append((score, so_far + [k], v.tails))
        
        tmp2 = sorted(tmp, reverse=True)
        
        for item in tmp2[:width]:
            node_queue.append(item)

    return sorted([(s, ''.join(item[:-1])) for item, s in zip(topn, scores)], reverse=True)


# In[58]:


get_ipython().run_cell_magic('time', '', 'predict("vade", trie)')


# In[61]:


get_ipython().run_cell_magic('time', '', 'predict("le", trie, n=10)')


# In[ ]:




