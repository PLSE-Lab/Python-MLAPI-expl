#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import re
import time

from gensim.models import Word2Vec
from tqdm import tqdm

tqdm.pandas()


# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[ ]:


sentences = pd.concat([df_train['title'], df_test['title']],axis=0)
train_sentences = list(sentences.progress_apply(str.split).values)


# In[ ]:


# Reference : https://www.quora.com/How-do-I-determine-Word2Vec-parameters
start_time = time.time()

model = Word2Vec(sentences=train_sentences, 
                 sg=1, 
                 size=250,
                 workers=4)

print(f'Time taken : {(time.time() - start_time) / 60:.2f} mins')


# In[ ]:


model.wv.save_word2vec_format('custom_glove_250d_no_processing.txt')


# In[ ]:




