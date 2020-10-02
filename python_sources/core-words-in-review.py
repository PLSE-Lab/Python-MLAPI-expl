#!/usr/bin/env python
# coding: utf-8

# In[4]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import re
import jieba
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## data overview

# In[5]:


path = "../input/Womens Clothing E-Commerce Reviews.csv"
df = pd.read_csv(path)
df.describe()


# ## ignore null value of text field
# when parsing "review text" field, it occurs errors in python due to the null value of text field

# In[6]:


clear_df = df.dropna(subset=["Review Text"])
clear_df.describe()


# ## Tokenize
# simple process, reserve only digits and letters , and words length larger than 1

# In[7]:


clear_df['tokens'] = clear_df['Review Text'].apply(
    lambda x: " ".join(
        [t for t in filter(lambda xx: xx and len(xx)>1, re.sub("[^0-9a-z]", " ", x.lower()).split(' '))]
    )
)


# In[8]:


clear_df['tokens'] ## check data


# In[9]:


from sklearn.feature_extraction.text import CountVectorizer
corpus = clear_df['tokens']
vectorizer = CountVectorizer()  
X = vectorizer.fit_transform(corpus)  
word = vectorizer.get_feature_names() ## the whole words in corpus 


# ## calculate tfidf as weight matrix

# In[10]:


from sklearn.feature_extraction.text import TfidfTransformer  
transformer = TfidfTransformer()  
tfidf = transformer.fit_transform(X)  
weight = tfidf.toarray()


# ## extract core words
# + use min_weight(tfidf score)
# + speed up via vectorization (np.applay_along_axis)

# In[ ]:


weight.shape
min_weight = 0.1
core_words = np.apply_along_axis(lambda vec: " ".join([t[0] for t in filter(lambda x: x[1]>min_weight, [(word[j], ele) for j,ele in enumerate(vec)])]), 1, weight)


# ## concat origin clean df

# In[ ]:


core_word_df = pd.DataFrame(core_words, columns=["CoreWords"])
core_word_df.shape
final = clear_df.join(core_word_df)


# In[ ]:


final[['Title', 'Review Text','tokens','CoreWords']]

