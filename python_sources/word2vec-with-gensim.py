#!/usr/bin/env python
# coding: utf-8

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


get_ipython().system('pip install gensim')


# In[ ]:


from gensim.models import Word2Vec, KeyedVectors
import pandas as pd
import nltk


# In[ ]:


df = pd.read_csv('../input/worldnews-on-reddit/reddit_worldnews_start_to_2016-11-22.csv')


# In[ ]:


df.head(10)


# In[ ]:


newsTitles = df['title'].values


# In[ ]:


newsTitles


# In[ ]:


nltk.download('punkt')


# In[ ]:


newsVec = [nltk.word_tokenize(title) for title in newsTitles]


# In[ ]:


newsVec


# In[ ]:


model = Word2Vec(newsVec, min_count = 1, size = 32)


# In[ ]:


model.most_similar('man')


# In[ ]:


vec = model['king'] - model['man'] + model['woman']
model.most_similar([vec])


# In[ ]:


# You can also load the Google's Word2Vec Pretrained 


# In[ ]:


#https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit


# In[ ]:


model = KeyedVectors.load_word2vec_format('../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin.gz', binary = True, limit = 100000)


# In[ ]:


vec = model['king'] - model['man'] + model['woman']
print(vec)


# In[ ]:


model.most_similar([vec])


# In[ ]:


vec = model['Germany'] - model['Berlin'] + model['Paris']


# In[ ]:


model.most_similar([vec])
# The most similar to Paris is France


# In[ ]:


vec = model["Cristiano_Ronaldo"] - model["football"] + model["tennis"]


# In[ ]:


model.most_similar([vec])
# You can see Nadal is in the list of similars


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




