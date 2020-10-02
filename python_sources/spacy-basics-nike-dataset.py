#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import spacy
import seaborn as sns
import random


# In[ ]:


nlp = spacy.load('en')


# In[ ]:


tweets = pd.read_csv("../input/justdoit_tweets_2018_09_07_2.csv")


# In[ ]:


tweets.shape


# Picking some Random Tweets from the dataset.

# In[ ]:


text = tweets.tweet_full_text[random.sample(range(1,100),10)]
text


# In[ ]:


combined = str(text)
doc = nlp(combined)


# In[ ]:


for token in doc:
    print(token)


# In[ ]:


for token in doc:
    print(token.text,token.pos_)


# Extracting Sentences

# In[ ]:


sents = list(doc.sents)
sents


# Named Entity Recognition

# In[ ]:


for entity in doc.ents:
    print(entity.text,entity.label_)


# In[ ]:


#Entity style Display
spacy.displacy.render(doc, style='ent',jupyter=True)


# In[ ]:


#Lemmatization
for token in doc:
    print(token.text,token.lemma_)


# In[ ]:


#Dependency Parser style Display
spacy.displacy.render(doc, style='dep',jupyter=True)


# In[ ]:




