#!/usr/bin/env python
# coding: utf-8

# # Fast GloVe data import (multiprocessing)
# 
# While trying to improve [the great LSTM baseline kernel](https://www.kaggle.com/jhoward/improved-lstm-baseline-glove-dropout-lb-0-048) by Jeremy Howard, I've noticed that a large chunk of my script exec time was spent on reading GloVe embeddings data into memory, especially when using the 840B 300d version.   
# Thankfully, this problem is quite easy to solve with Python's `multiprocessing` module. 

# In[ ]:


import numpy as np
from multiprocessing import Pool

num_cpu = 4
embed_size = 300
glove_file_path = '../input/glove840b300dtxt/glove.840B.300d.txt'


# In[ ]:


def get_coefs(row):
    row = row.strip().split()
    # can't use row[0], row[1:] split because 840B contains multi-part words 
    word, arr = " ".join(row[:-embed_size]), row[-embed_size:]
    return word, np.asarray(arr, dtype='float32')


# In[ ]:


def get_glove():
    return dict(get_coefs(row) for row in open(glove_file_path))


# In[ ]:


def get_glove_fast():
    pool = Pool(num_cpu)
    with open(glove_file_path) as glove_file:
        return dict(pool.map(get_coefs, glove_file, num_cpu))


# In[ ]:


# Time for sequential data import
get_ipython().run_line_magic('time', 'glove1 = get_glove()')


# In[ ]:


# Time for multiprocessing data import
get_ipython().run_line_magic('time', 'glove2 = get_glove_fast()')


# In[ ]:


assert len(glove1) == len(glove2)

