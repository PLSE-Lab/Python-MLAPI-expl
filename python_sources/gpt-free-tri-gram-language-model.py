#!/usr/bin/env python
# coding: utf-8

# ## Exercise | Building your own language model

# ---

# ### A quick recap to language models
# 
# #### To deep dive and to go Zero-to-Hero in NLP, check out this awesome github repo. Link below
# [Zero-to-Hero in NLP](https://github.com/samacker77/Zero-to-Hero-in-NLP)
# 
# > Language models assign probability to predict the next word based on the previous words.
# 
# 1. n-gram language model - Predicts next word by looking at (n-1) words in the past.
#     - bi-gram/2-gram language model - Predicts next word by looking at (2-1) = 1 word in the past.
#     - tri-gram/3-gram language model - Predicts next word by looking at (3-1) = 2 words in the past.
#     
#     
# 2. How to assign probabilites?
# 
# Suppose the past words for a tri-gram language model are $w_1$, $w_2$ and we need to assign probability for the next word $w$.
# 
# Count the number of times $w_1$ and $w_2$ come together[C1] , and count the number of times $w_1$, $w_2$, and $w$ come together [C2].
# 
# Divide C2 by C1.
# 
# $P(w | w_1,w_2) = \frac{C2}{C1}$
# 
# 
# For example,
# 
# Imagine you have a corpus, and you want to assign probability to the word 'Sam' after seeing two words 'I am'.
# 
# Count the number of times 'I am Sam' come together, and count the number of times 'I am' come together. You will get the probability of getting 'Sam' after seeing 'I am'. Easy maths right?
# 
# To generate more coherent sentence, we continue this simple math with chain rule of probability.

# ---
# 

# In[ ]:


import importlib
import nltk
nltk.download('brown')


# ## Python code
# 
# ---

# In[ ]:


# Get a dataset

from nltk.corpus import brown

# Import APIs for generating trigrams

from nltk import trigrams

# Import Counter API to store frequencies

from collections import Counter, defaultdict


# ### Let's name our State-Of-The-Art model as gptFree
# 
# 

# In[ ]:


gptFree = defaultdict(lambda: defaultdict(lambda: 0))

# We don't want a KeyError, hence using defaultdict. 
# It will assign zero probability if a trigram probobality turns out zero

print(gptFree)


# ### Let's count and store frequency of trigrams in the reuters data set.

# In[ ]:


for sentence in brown.sents():
    for word_1, word_2, word_3 in trigrams(sentence, pad_right=True, pad_left=True):
        gptFree[(word_1, word_2)][word_3] += 1 # Storing frequencies as and updating +1 as we see them
        
        


# ### Convert frequencies to probabilites

# In[ ]:


for word1_word2 in gptFree:
    freq = float(sum(gptFree[word1_word2].values())) # Fetch frequencies of two words coming together
    for word_3 in gptFree[word1_word2]:
        gptFree[word1_word2][word_3] /= freq


# ### Predict next word
# 
# 

# > Provide two starter words to predict the next word

# In[ ]:


dict(gptFree['I','am'])


# In[ ]:


dict(gptFree['What','is'])


# In[ ]:


dict(gptFree['The','bank'])


# In[ ]:




