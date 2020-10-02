#!/usr/bin/env python
# coding: utf-8

# This is part of a course project on NLP and deep learning (CS224n at Stanford). We aim to leverage principles of NLP to consistently predict tags of Stack Exchange posts across disciplines.

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


df = pd.read_csv("../input/cooking.csv")
df.head()


# ## Data preprocessing
# 
# * Convert tags into lists
# * Remove html tags

# In[ ]:


df["tags"] = df["tags"].map(lambda x: x.split())


# In[ ]:


print(df.loc[10])


# In[ ]:


print(df.iloc[10])


# ## Feature Creation
# First test out bag-of-words approach to vectorize posts.

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer = "word",                                tokenizer = None,                                 preprocessor = None,                              stop_words = None,                                max_features = 5000) 


# In[ ]:


num_posts = df["content"].size
posts = []
for i in range( 0, num_posts ):
    posts.append( df["content"][i] )


# In[ ]:


train_data_features = vectorizer.fit_transform(posts)
train_data_features = train_data_features.toarray()

print(train_data_features.shape)


# First implementation of latent dirichlet allocation algorithm.

# In[ ]:


type(df.loc[0:5])


# In[ ]:


from gensim import corpora, models

dictionary = corpora.Dictionary(df.loc[0:5])


# ## References:
# 
# [1] https://www.kaggle.com/l3nnys/transfer-learning-on-stack-exchange-tags/useful-text-preprocessing-on-the-datasets
# 
# [2] https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words
# 
# [3] https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html
# 
# [4] http://engineering.flipboard.com/2017/02/storyclustering?from=groupmessage&isappinstalled=0

# In[ ]:




