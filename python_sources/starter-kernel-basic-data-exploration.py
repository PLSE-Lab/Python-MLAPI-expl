#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


# # Load the recipe data

# In[ ]:


df = pd.read_json("../input/recipes.json")


# In[ ]:


df.head()


# # Basic dataset statisitcs

# In[ ]:


print("We have {} recipes".format(df.shape[0]))


# We use spacy to tokenize the Instructions and investigate the vocabulary.

# In[ ]:


get_ipython().system('python -m spacy download de_core_news_sm')


# In[ ]:


import spacy
nlp = spacy.load('de_core_news_sm', disable=['parser', 'tagger', 'ner'])


# In[ ]:


tokenized = [nlp(t) for t in df['Instructions'].values]


# In[ ]:


for t in tokenized[0]:
    print(t)


# In[ ]:


vocab = {}
for txt in tokenized:
    for token in txt:
        if token.text not in vocab.keys():
            vocab[token.text] = len(vocab)


# In[ ]:


print("Number of unique tokens: {}".format(len(vocab)))


# In[ ]:




