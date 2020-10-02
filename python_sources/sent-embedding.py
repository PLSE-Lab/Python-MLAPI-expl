#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np, pandas as pd
import json
import ast 
from textblob import TextBlob
import nltk
import torch
import pickle
from scipy import spatial
import warnings
warnings.filterwarnings('ignore')
import spacy
from nltk import Tree
en_nlp = spacy.load('en')
from nltk.stem.lancaster import LancasterStemmer
st = LancasterStemmer()
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer


# In[2]:


# !conda update pandas --y


# In[3]:


train = pd.read_csv("../input/sentence-embedding/train.csv",encoding="latin-1")


# In[4]:


train.shape


# ### Loading Embedding dictionary

# In[5]:


with open("../input/sent-embedding/dict_embeddings1.pickle", "rb") as f:
    d1 = pickle.load(f)


# In[ ]:





# In[6]:


with open("../input/sent-embedding/dict_embeddings2.pickle", "rb") as f:
    d2 = pickle.load(f)


# In[7]:


dict_emb = dict(d1)
dict_emb.update(d2)


# In[8]:


len(dict_emb)


# In[9]:


del d1, d2


# ## Data Processing

# In[10]:


def get_target(x):
    idx = -1
    for i in range(len(x["sentences"])):
        if x["text"] in x["sentences"][i]: idx = i
    return idx


# In[11]:


train.head(3)


# In[12]:


train.shape


# In[13]:


train.dropna(inplace=True)


# In[14]:


train.shape


# In[15]:


def process_data(train):
    
    print("step 1")
    train['sentences'] = train['context'].apply(lambda x: [item.raw for item in TextBlob(x).sentences])
    
    print("step 2")
    train["target"] = train.apply(get_target, axis = 1)
    
    print("step 3")
    train['sent_emb'] = train['sentences'].apply(lambda x: [dict_emb[item][0] if item in                                                           dict_emb else np.zeros(4096) for item in x])
    print("step 4")
    train['quest_emb'] = train['question'].apply(lambda x: dict_emb[x] if x in dict_emb else np.zeros(4096) )
        
    return train   


# In[16]:


train = process_data(train)


# In[17]:


train.head(3)


# In[25]:


train.to_csv("procesed_train.csv")


# In[26]:


del train


# In[27]:


import gc
gc.collect()


# In[ ]:




