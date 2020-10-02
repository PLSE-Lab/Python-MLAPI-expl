#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' pip install git+https://github.com/LIAAD/yake')
    
import yake


# In[ ]:


import pandas as pd


# In[ ]:


forum_posts = pd.read_csv("../input/ForumMessages.csv")


# In[ ]:


# take keywords for each post & save in list
simple_kwextractor = yake.KeywordExtractor()

# create empty list to save our keywords to
sentences = []

# subsample forum posts
sample_posts = forum_posts.Message[:10]

# loop through forum posts & extract keywords
for post in sample_posts:
    post_keywords = simple_kwextractor.extract_keywords(post)
    
    sentence_output = ""
    for word, number in post_keywords:
        sentence_output += word + " "
    
    sentences.append(sentence_output)


# In[ ]:


# original post
sample_posts[2]


# In[ ]:


# just the keywords from that post
sentences[2]

