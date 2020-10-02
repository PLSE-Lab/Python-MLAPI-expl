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
# 
# ### Install the Kaggle API and download the datasets

# ## Model with TF-IDF and Ranfom Forest

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
import string
from joblib import Parallel, delayed
from tqdm import tqdm_notebook as tqdm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
stop_words = set(stopwords.words('english'))
stem = SnowballStemmer('english')


# In[ ]:


train_df = pd.read_csv("../input/train.csv")
train_df = train_df[['id','comment_text', 'target']]
test_df = pd.read_csv("../input/test.csv")


# In[ ]:


train_df.head()


# In[ ]:


train_df.target.hist()


# In[ ]:


train_df.shape


# In[ ]:





# In[ ]:


test_df.head()


# In[ ]:


test_df.shape


# In[ ]:


train_df = train_df.sample(100000, random_state=42)


# In[ ]:


train_df.shape


# Create tokens

# In[ ]:


def tokenize(text):
    
    tokens = []
    for token in word_tokenize(text):
        if token in string.punctuation: continue
        if token in stop_words: continue
        tokens.append(stem.stem(token))
    
    return " ".join(tokens)


# In[ ]:


train_tokens = Parallel(n_jobs=-1, verbose=1)(delayed(tokenize)(text) for text in train_df['comment_text'].tolist())


# In[ ]:


train_tokens[0]


# In[ ]:


test_tokens = Parallel(n_jobs=-1, verbose=1)(delayed(tokenize)(text) for text in test_df['comment_text'].tolist())


# In[ ]:


len(train_tokens + test_tokens)


# In[ ]:


vect = TfidfVectorizer()
vect.fit(train_tokens + test_tokens)


# In[ ]:


X = vect.transform(train_tokens)
y = train_df['target']


# In[ ]:


reg = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42, max_depth=10)
reg.fit(X, y)


# In[ ]:


test_X =  vect.transform(test_tokens)
test_y = reg.predict(test_X)


# In[ ]:


submisson_df = pd.read_csv("../input/sample_submission.csv")
submisson_df['prediction'] = test_y
submisson_df['prediction'] = submisson_df['prediction'].apply(lambda x: "%.5f" % x if x > 0 else 0.0)


# In[ ]:


submisson_df.to_csv("submission.csv", index=False)

