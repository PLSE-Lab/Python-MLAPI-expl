#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from tqdm import tqdm
pd.set_option("display.max_colwidth", 200)


# In[ ]:


# read the data
reviews_df = pd.read_csv('../input/Womens Clothing E-Commerce Reviews.csv')


# In[ ]:


# Let's check out the missing values
reviews_df.isnull().sum()


# In[ ]:


# drop rows with NA
reviews_df.dropna(subset=['Review Text', 'Division Name'], inplace=True)


# In[ ]:


reviews_df['Rating'].value_counts()


# We have 5-scale rating system.

# In[ ]:


# clean reviews, removing everything except alphabets
reviews_df['Review_Tidy'] = reviews_df['Review Text'].str.replace("[^a-zA-Z#]", " ")

# removing tokens having length 1
reviews_df['Review_Tidy'] = reviews_df['Review_Tidy'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>1]))


# In[ ]:


# tokenizing reviews 
tokenized_reviews = reviews_df['Review_Tidy'].apply(lambda x: x.split())
tokenized_reviews = tokenized_reviews.tolist()


# ## Feature Extraction
# 
# Here I am using word2vec to extract numerical features from the tokenized reviews text.

# In[ ]:


import gensim

# using word2vec to extract features aka word vectors
model_w2v = gensim.models.Word2Vec(
            tokenized_reviews,
            size=200, # desired no. of features/independent variables 
            window=5, # context window size
            sg = 1, # 1 for skip-gram model
            hs = 0,
            negative = 10, # for negative sampling
            workers= 4, # no.of cores
            seed = 34)


# Let's see the power of this  word2vec model. I'll give a word as input and it will throw a list of most similar words.

# In[ ]:


model_w2v.wv.most_similar(positive="jeans")


# In[ ]:




