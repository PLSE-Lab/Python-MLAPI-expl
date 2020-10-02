#!/usr/bin/env python
# coding: utf-8

# ## The following code snippets are meant to offer some examples of analysis of the handles and texts in the tweets

# In[ ]:


import pandas as pd


# In[ ]:


tweetstable = pd.read_csv("../input/saarc87964.csv", encoding = "Latin-1")


# In[ ]:


tweetstable[0:5]


# In[ ]:


len(tweetstable.Handle)


# In[ ]:


handles = []
for handle in tweetstable.Handle:
    handles.append(handle)


# In[ ]:


import nltk


# In[ ]:


hfdist = nltk.FreqDist(handles)


# In[ ]:


hfdist.most_common(1000)


# In[ ]:


len(tweetstable.Tweet)


# In[ ]:


tfdist = nltk.FreqDist(x for x in tweetstable.Tweet)


# In[ ]:


tfdist.most_common(1000)


# In[ ]:


dfdist = nltk.FreqDist(x for x in tweetstable.Day)


# In[ ]:


dfdist.most_common(1000)


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


hfdist.plot(25)


# In[ ]:


dfdist.plot(25)


# In[ ]:


hashes = []
for tweet in tweetstable.Tweet:
    tokens = str(tweet).split(" ")
    for tok in tokens:
        if "#" in tok:
            hashes.append(tok)


# In[ ]:


hashdist = nltk.FreqDist(hash.lower() for hash in hashes)


# In[ ]:


hashdist.most_common(1000)

