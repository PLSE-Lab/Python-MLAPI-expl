#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import nltk
nltk.download()


# In[ ]:


from nltk.book import *


# In[ ]:


len(text3)


# In[ ]:


len(set(text3))


# In[ ]:


text3.count("In")


# In[ ]:


FreqDist(text3).most_common(50)


# In[ ]:


text2.collocations()


# In[ ]:


## Tokenization

from nltk import word_tokenize

import nltk, re, pprint


# In[ ]:


from urllib import request
url = "http://www.gutenberg.org/files/2554/2554-0.txt"
response = request.urlopen(url)
raw = response.read().decode('utf8')


# In[ ]:


len(raw)


# In[ ]:


tokens = word_tokenize(raw)


# In[ ]:


len(tokens)


# In[ ]:


tokens[100:150]


# In[ ]:


text = nltk.Text(tokens)


# In[ ]:


bigrm = list(nltk.bigrams(text))


# In[ ]:


print(*map(' '.join, bigrm), sep=', ')


# In[ ]:




