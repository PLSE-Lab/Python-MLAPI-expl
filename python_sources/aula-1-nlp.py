#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output. 


# In[ ]:


get_ipython().system('python -m spacy download pt_core_news_sm')

import spacy


# In[ ]:


nlp=spacy.load('pt_core_news_sm')


# In[ ]:


file=open(u'/kaggle/input/data.txt')

data=file.read()

file.close()


# In[ ]:


doc=nlp(data)


# In[ ]:


for token in doc:
    print(f'{token.text:{10}} {token.pos_:{5}} {token.dep_:{5}}')


# In[ ]:


nlp.pipeline
nlp.pipe_names


# In[ ]:


ent_list=[(token.text, token.pos_) for token in doc]
ent_list


# In[ ]:


for sent in doc.sents: 
    print(sent)

