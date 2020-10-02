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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# ## From Starters

# In[ ]:


#Load required libraries
import numpy as np
import pandas as pd
#For displaying complete rows info
pd.options.display.max_colwidth=500
import tensorflow as tf
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
print(tf.__version__)


# 

# ### Sumamry Followed by Article

# In[ ]:


#Load data into pandas dataframe
df=pd.read_csv("../input/articles.csv",encoding="utf8")


# In[ ]:


df.head(2)


# In[ ]:


print(df["title"][0],"\n",df["text"][0])


# In[ ]:


#Properly formatted data removing nans
df.drop_duplicates(subset=["text"],inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)


# ### WORD2VEC MODEL USING GENSIM

# In[ ]:


import gensim
import string
import re


# In[ ]:


articles_tokens=[]
for i in range(len(df["text"])):
    articles_tokens.append([word for word in word_tokenize(str(df["text"][i].lower())) if len(word)>2])


# In[ ]:


model = gensim.models.Word2Vec(articles_tokens, min_count=5,size=100,workers=4)


# In[ ]:


model.wv.most_similar("lula")


# In[ ]:





# In[ ]:


model.wv.most_similar("propina")


# In[ ]:


model.wv.most_similar("esporte")


# In[ ]:




