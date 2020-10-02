#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt # plotting
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
from collections import Counter
from collections import deque
import random


# In[ ]:


get_ipython().system('pip install glove_python')


# In[ ]:


print(os.listdir('../input/datagrand/datagrand/'))


# In[ ]:


sentences = []           
with open('../input/datagrand/datagrand/corpus.txt') as f:
    lines = f.readlines()
    for sentence in tqdm(lines):
        #print(sentence)
        sentence = sentence.replace('\n','').split('_')
        if len(sentence) > 0:
            sentences.append(sentence)


# In[ ]:


special_words = []
with open('../input/datagrand/datagrand/train.txt') as f:
    lines = f.readlines()
    for sentence in tqdm(lines):
        word_list = []
        words = sentence.replace('\n','').split('  ')
        for word in words:
            split_word = word.split('/')
            word_meta = split_word[0]
            word_meta_split = word_meta.split('_')
            word_list += word_meta_split
        if len(word_list)>1:
            sentences.append(word_list)
            sentences.append(word_list)
        else:
            special_words.append(word_list)


# In[ ]:


from glove import Glove
from glove import Corpus


# In[ ]:


corpus_model = Corpus()
corpus_model.fit(sentences, window=10)


# In[ ]:


glove = Glove(no_components=256, learning_rate=0.05,max_count=100,random_state=2019)
glove.fit(corpus_model.matrix, epochs=50,
          no_threads=1, verbose=True)
glove.add_dictionary(corpus_model.dictionary)


# In[ ]:



glove.save('glove.model')


# In[ ]:





# In[ ]:





# In[ ]:




