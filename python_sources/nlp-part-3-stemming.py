#!/usr/bin/env python
# coding: utf-8

# # Stemming

# 
# PorterStemmer for Stemming, we import this from nltk.stem

# word_tokenize from nltk to tokenize the sentence into words

# In[ ]:


from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize


# Object named obj for PorterStemmer

# In[ ]:


obj = PorterStemmer()


# 
# Define list of words for which we want to perform stemming

# In[ ]:


words = ["play","player","playing","played"]


# Calling stem on list referenced by w

# In[ ]:


for w in words: 
    print(w, " : ",obj.stem(w))


# # End of notebook

# In[ ]:




