#!/usr/bin/env python
# coding: utf-8

# # Spooky Author Identification
# Attempting to identify the writing of Edgar Allan Poe, H.P. Lovecraft, and Mary Shelley from short samples of writing using Python.

# In[1]:


import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
import random
import nltk


# > ## Sample Submission

# In[2]:


sample_sub = pd.read_csv("../input/sample_submission.csv")
sample_sub.head(5)


# # 1. A Quick Look at the Data
# In this section we import the data and take a look at the columns and entries we have to work with.

# In[3]:


train = pd.read_csv("../input/train.csv")
train.head(5)


# In[4]:


test = pd.read_csv("../input/test.csv")
test.head(5)


# ## Splitting Up The Training Set

# In[5]:


eap = train.loc[train.author == "EAP"]
hpl = train.loc[train.author == "HPL"]
mws = train.loc[train.author == "MWS"]


# # 2. A Perfectly Random Model
# Before we embark on implementing complicated language processing techniques to try to determine who authored each piece of text, considering that we only have three options to choose from, we could first try to implement a perfectly random model as a base comparison for future models.

# In[6]:


results = pd.DataFrame()
results["id"] = test.id
results["EAP"] = 0
results["HPL"] = 0
results["MWS"] = 0


# In[7]:


for i in results.T:
    r = random.randint(1, 3)
    if r == 1:
        results.at[i, "EAP"] = 1
    if r == 2:
        results.at[i, "HPL"] = 1
    if r == 3:
        results.at[i, "MWS"] = 1

results.head(5)


# In[8]:


# Kaggle Score: 15.77882
# results.to_csv("../results/random.csv")


# # 3. Using NLTK

# In[10]:


eap_words = []
hpl_words = []
mws_words = []

for i in train.T:
    if train.author[i] == "EAP":
        eap_words.append(nltk.word_tokenize(train.text[i]))
    if train.author[i] == "HPL":
        hpl_words.append(nltk.word_tokenize(train.text[i]))
    if train.author[i] == "MWS":
        mws_words.append(nltk.word_tokenize(train.text[i]))


# In[11]:


stopwords = nltk.corpus.stopwords.words('english')


# In[12]:


eap_words = [for word in wl in eap_words]
    
for phrase in hpl_words:
    phrase = [word for word in phrase if word.lower() not in stopwords]

for phrase in mws_words:
    phrase = [word for word in phrase if word.lower() not in stopwords]


# In[73]:


print(eap_words[0:4])
#print(hpl_words[0:4])
#print(mws_words[0:4])


# In[ ]:




