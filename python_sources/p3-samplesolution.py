#!/usr/bin/env python
# coding: utf-8

# # Project 3
# 
# 
# # Conversations Toxicity Detection
# 
# Jigsaw Unintended Bias in Toxicity Classification 
# 
# Detect toxicity across a diverse range of conversations
# 
# 
# https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data#
# 
# # Sample Solution

# ## Model with TF-IDF and Ranfom Forest

# In[2]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
import string
from joblib import Parallel, delayed
from tqdm import tqdm_notebook as tqdm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
stop_words = set(stopwords.words('english'))
stem = SnowballStemmer('english')


# In[5]:


train_df = pd.read_csv("../input/train.csv")
train_df = train_df[['id','comment_text', 'target']]
test_df = pd.read_csv("../input/test.csv")


# In[6]:


train_df.head()


# In[7]:


train_df.target.hist()


# In[8]:


train_df.shape


# In[9]:


test_df.head()


# In[10]:


test_df.shape


# In[11]:


# train_df = train_df.sample(100000, random_state=42)


# In[12]:


train_df.shape


# Create tokens

# In[13]:


def tokenize(text):
    
    tokens = []
    for token in word_tokenize(text):
        if token in string.punctuation: continue
        if token in stop_words: continue
        tokens.append(stem.stem(token))
    
    return " ".join(tokens)


# In[14]:


train_tokens = Parallel(n_jobs=-1, verbose=1)(delayed(tokenize)(text) for text in train_df['comment_text'].tolist())


# In[15]:


train_tokens[0]


# In[16]:


test_tokens = Parallel(n_jobs=-1, verbose=1)(delayed(tokenize)(text) for text in test_df['comment_text'].tolist())


# In[17]:


len(train_tokens + test_tokens)


# In[18]:


vect = TfidfVectorizer()
vect.fit(train_tokens + test_tokens)


# In[19]:


X = vect.transform(train_tokens)
y = train_df['target']


# In[20]:


svr = LinearSVR(random_state=71, tol=1e-3, C=1.2)
svr.fit(X, y)


# In[21]:


test_X =  vect.transform(test_tokens)
test_y = svr.predict(test_X)


# In[22]:


submisson_df = pd.read_csv("../input/sample_submission.csv")
submisson_df['prediction'] = test_y
submisson_df['prediction'] = submisson_df['prediction'].apply(lambda x: "%.5f" % x if x > 0 else 0.0)


# In[23]:


submisson_df.to_csv("submission.csv", index=False)


# In[ ]:




