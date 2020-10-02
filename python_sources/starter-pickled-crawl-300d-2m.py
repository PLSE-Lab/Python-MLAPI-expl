#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# Loading FT Crawl from .txt files typically takes +3min on Kernels:

# In[1]:


# FT_EMBEDDING_PATH = '../input/fasttextcrawl300d2m/crawl-300d-2m.vec'

# def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

# def load_embeddings(embed_dir):
#     embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in tqdm(open(embed_dir)))
#     return embedding_index

# ft = load_embeddings(FT_EMBEDDING_PATH)


# Now, you can load from the pickled file directly. It loads the entire embedding, so if you only wish to use a subset, be sure to `del ftext` and `gc.collect()` once you're done with it:

# In[3]:


import pickle
from time import time

t = time()
with open('../input/crawl-300d-2M.pkl', 'rb') as fp:
    ftext = pickle.load(fp)
print(time()-t)


# In[4]:


len(ftext)


# In[6]:


list(ftext.keys())[1]


# In[7]:


ftext[',']


# In[ ]:


Happy Kaggling


# 
