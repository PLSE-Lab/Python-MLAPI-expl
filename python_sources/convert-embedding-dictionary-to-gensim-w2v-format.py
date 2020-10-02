#!/usr/bin/env python
# coding: utf-8

# This kernel converts a Python dict of word embedding vectors into gensim's word2vec binary format. 
# 
# This leads to faster loading and you can do pretty stuffs with gensim.
# 
# Source:
# [scikit learn \- Convert Python dictionary to Word2Vec object \- Stack Overflow](https://stackoverflow.com/questions/45981305/convert-python-dictionary-to-word2vec-object)
# 

# In[ ]:


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


# In[ ]:


import gensim
from tqdm import tqdm_notebook as tqdm


# # Load embedding in a usual way

# In[ ]:


def get_coefs(word, *arr):
    return word, np.asarray(arr[:-1], dtype='float32')

def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.split(' ')) for line in f)


# In[ ]:


CRAWL_EMBEDDING_PATH = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'


# In[ ]:


get_ipython().run_cell_magic('time', '', 'crawl_emb_dict = load_embeddings(CRAWL_EMBEDDING_PATH)')


# In[ ]:


len(crawl_emb_dict)


# In[ ]:


crawl_emb_dict['kaggle'].shape


# In[ ]:


crawl_emb_dict['kaggle']


# # Save it in gensim w2v binary format

# In[ ]:


def save_word2vec_format(fname, vocab, vector_size, binary=True):
    """Store the input-hidden weight matrix in the same format used by the original
    C word2vec-tool, for compatibility.

    Parameters
    ----------
    fname : str
        The file path used to save the vectors in.
    vocab : dict
        The vocabulary of words.
    vector_size : int
        The number of dimensions of word vectors.
    binary : bool, optional
        If True, the data wil be saved in binary word2vec format, else it will be saved in plain text.


    """
    
    total_vec = len(vocab)
    with gensim.utils.smart_open(fname, 'wb') as fout:
        print(total_vec, vector_size)
        fout.write(gensim.utils.to_utf8("%s %s\n" % (total_vec, vector_size)))
        # store in sorted order: most frequent words at the top
        for word, row in tqdm(vocab.items()):
            if binary:
                row = row.astype(np.float32)
                fout.write(gensim.utils.to_utf8(word) + b" " + row.tostring())
            else:
                fout.write(gensim.utils.to_utf8("%s %s\n" % (word, ' '.join(repr(val) for val in row))))


# In[1]:


with open('dummy_text_to_prevent_kernel_page_loding_from_being_heavy.txt', 'w') as f:
    f.write(':3')
    


# In[ ]:


get_ipython().run_cell_magic('time', '', "save_word2vec_format(binary=True, fname='crawl-300d-2M.bin', vocab=crawl_emb_dict, vector_size=300)")


# In[ ]:


with open('dummy_text_to_prevent_kernel_page_loding_from_being_heavy2.txt', 'w') as f:
    f.write(':9')


# # Load it again

# In[ ]:


get_ipython().run_cell_magic('time', '', "model = gensim.models.KeyedVectors.load_word2vec_format('crawl-300d-2M.bin', binary=True)")


# In[ ]:


model['kaggle']


# In[ ]:


get_ipython().run_cell_magic('time', '', "model.most_similar('kaggle')")


# In[ ]:




