#!/usr/bin/env python
# coding: utf-8

# This kernel is inspired from the extended word-to-word analysis of Deep_patience (https://www.kaggle.com/aisaactirona/cudnnlstm-cudnngru). I just wanted to point out that some of the words that could pertain to toxic sentences, do not necessarily do so. I give some example below, however more words can be found in a more extended search.
# 

# In[31]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os 


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# Lets keep only Comments and their toxicity 'level'

# In[ ]:


comment_only = train[['comment_text', 'target']]


# Example Word 1: 'Sex' 

# In[ ]:


sex_comments = comment_only[comment_only['comment_text'].str.contains('sex')]
print("Mean toxicity of word 'Sex': {}".format(np.round(sex_comments['target'].mean(),2)))


# In[ ]:


plt.hist(sex_comments['target'])
plt.show()


# Example Word 2: 'damn' 

# In[ ]:


damn_comments = comment_only[comment_only['comment_text'].str.contains('damn')]
print("Mean toxicity of word 'damn': {}".format(np.round(damn_comments['target'].mean(),2)))


# In[ ]:


plt.hist(damn_comments['target'])
plt.show()


# In[ ]:


god_comments = comment_only[comment_only['comment_text'].str.contains('God')]
print("Mean toxicity of word 'God': {}".format(np.round(god_comments['target'].mean(),2)))


# In[ ]:


plt.hist(god_comments['target'])
plt.show()


# In[ ]:


porn_comments = comment_only[comment_only['comment_text'].str.contains('porn')]
print("Mean toxicity of word 'porn': {}".format(np.round(porn_comments['target'].mean(),2)))


# In[ ]:


plt.hist(porn_comments['target'])
plt.show()


# Lets also include something sure toxic for sanity check :)

# In[ ]:


asshole_comments = comment_only[comment_only['comment_text'].str.contains('asshole')]
print("Mean toxicity of word 'asshole': {}".format(np.round(asshole_comments['target'].mean(),2)))


# Right, so that concludes this quick analysis on bias on toxic terms. I hope it helps for model tuning!
