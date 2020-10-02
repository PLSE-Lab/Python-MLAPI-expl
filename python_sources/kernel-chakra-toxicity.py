#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


df_train = pd.read_csv("../input/train.csv")


# In[21]:


## Starting with EDA!!!


# In[3]:


df_train.head(5)


# In[4]:


df_train.describe()


# In[5]:


df_train.columns


# In[6]:


text_features = df_train['comment_text']


# In[12]:


text_features.tolist()[0].split()


# In[15]:


# Import NLTK libraries to derive usefulness from the comments column
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# In[16]:


# Set Stopwords for English Lnguage
stop_words = set(stopwords.words('english'))


# In[17]:


word_tokens = word_tokenize(text_features.tolist()[0])


# In[18]:


word_tokens


# In[19]:


filtered_sentence = [w for w in word_tokens if not w in stop_words]


# In[20]:


filtered_sentence


# In[ ]:




