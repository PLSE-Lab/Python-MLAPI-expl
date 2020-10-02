#!/usr/bin/env python
# coding: utf-8

# #### Hello everyone , 
#  In this kernel we will go together into  ****Visualization**** Disaster Tweets data to learn how words related together using t-SNE natural language processing NLP techniques.
#  
# ** Before any thing , What is T-SNE ?
# ** 
# 
#  T-distributed Stochastic Neighbor Embedding (t-SNE) is a machine learning algorithm for visualization developed by Laurens van der Maaten and Geoffrey Hinton It is a nonlinear dimensionality reduction technique well-suited for embedding high-dimensional data for visualization in a low-dimensional space of two or three dimensions. Specifically, it models each high-dimensional object by a two- or three-dimensional point in such a way that similar objects are modeled by nearby points and dissimilar objects are modeled by distant points with high probability.
#  
#  [To read more](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)

# #### This kernel will be devided into the following parts
# 
# 1. Data Exploration
# 2. Data Preprocessing
# 3. Data Vizualization 

# Load libraries : 

# In[ ]:


import pandas as pd
pd.options.mode.chained_assignment = None 
import numpy as np
import re
import nltk

from gensim.models import word2vec

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
STOP_WORDS = nltk.corpus.stopwords.words()


# In[ ]:


get_ipython().system(' ls "../input/nlp-getting-started"')


# **1. Data Exploration**

# In[ ]:


train=pd.read_csv("../input/nlp-getting-started/train.csv")
test=pd.read_csv("../input/nlp-getting-started/test.csv")
submission=pd.read_csv("../input/nlp-getting-started/sample_submission.csv")


# In[ ]:


train.head()


# In[ ]:


print(train.shape)


# **2. Data Preprocessing**

# * clean nan values
# * delete columns not help in training or viusalize like id,location and keyword 
# * convert letters in text to lowercase 
# * remove numbers and symobls 
# *  take toknize of text 
# * Create Corpus of Training

# In[ ]:


def clean(text):
    regex = re.compile('([^\s\w]|_)+')
    sentence = regex.sub('', text).lower()
    sentence = sentence.split(" ")
    
    for word in list(sentence):
        if word in STOP_WORDS:
            sentence.remove(word)  
            
    sentence = " ".join(sentence)
    return sentence


# In[ ]:


train = train.dropna(axis=0)
train=train.reset_index()
for i in range (train.shape[0]):
    train.at[i,'text']=clean(train.loc[i,'text'])


# Create Corpus 

# In[ ]:


corpus=[]

for i in range(train.shape[0]):
    corpus.append(train['text'][i].split(" "))
    


# In[ ]:


print(corpus[:3])


# ## Word 2 Vec
# 
# Word2vec is a group of related models that are used to produce word embeddings. These models are shallow, two-layer neural networks that are trained to reconstruct linguistic contexts of words. Word2vec takes as its input a large corpus of text and produces a vector space
# 
# in this example have 100 dims

# In[ ]:


model = word2vec.Word2Vec(corpus, size=100, window=10, min_count=35, workers=4)


# In[ ]:


model.wv['news']


# In[ ]:


def drawing(model):
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


# In[ ]:


drawing(model)


# Can Using Word2Vec to get Simialr words 

# In[ ]:


model.most_similar('news')


# ## congratulation 
# > ## Happy End

# In[ ]:




