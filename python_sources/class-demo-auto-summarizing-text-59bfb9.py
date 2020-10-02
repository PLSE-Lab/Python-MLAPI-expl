#!/usr/bin/env python
# coding: utf-8

# # TL;DR - Automated Gist
# ## Find the most important words
# ### Word Importance = Word Frequency
# ## Compute a significance score for sentences based on words they contain
# ### Significant score = Sum(Word Importance)
# ## Pick the top most significant sentences
# 
# * Retrieve Text
# * Preprocess Text
# * Extract Sentences
# 
# #### Source: PluralSight - Natural Langauge Processing

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import accuracy_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from string import punctuation


# In[ ]:


def textAnalysis(articleURL):
#     articleURL="https://arstechnica.com/cars/2018/10/honda-will-use-gms-self-driving-technology-invest-2-75-billion/"
    response = requests.get(articleURL)
    response.encoding = 'utf-8'
    data = response.text
    soup = BeautifulSoup(data, features="html.parser")
    print(soup)


# In[ ]:


#In Kaggle I can't put all of the code under the if block while it's in different kernals or else i would. 
if soup.find('article').text:
    print("Article has article tag")
else: 
    print("Aerticle has no article tag")
   


# In[ ]:


text = ' '.join(map(lambda p: p.text, soup.find_all('article')))

text = text.replace("\n", " ")
text = text.replace("'s", "")
text = text.replace(".", ". ")
print(text)


# In[ ]:


text.encode('ascii', 'ignore')
print(text)
                                           


# In[ ]:


sents = sent_tokenize(text)
sents


# In[ ]:


word_sent = word_tokenize(text.lower())
word_sent


# In[ ]:


_stopwords = set(stopwords.words('english') + list(punctuation))
_stopwords


# In[ ]:


# Filter out stopword
word_sent=[word for word in word_sent if word not in _stopwords]
word_sent


# In[ ]:


from nltk.probability import FreqDist
freq = FreqDist(word_sent)
freq


# In[ ]:


from heapq import nlargest
nlargest(10, freq, key=freq.get)


# In[ ]:


# We want to create a signifcant score ordered by highest frequency
from collections import defaultdict
ranking = defaultdict(int)
for i,sent in enumerate(sents):
    for w in word_tokenize(sent.lower()):
        if w in freq:
            ranking[i] += freq[w]
ranking


# In[ ]:


# Top 4 Sentences
sents_idx = nlargest(3, ranking, key=ranking.get)
sents_idx


# In[ ]:


def main():
     textAnalysis("https://arstechnica.com/cars/2018/10/honda-will-use-gms-self-driving-technology-invest-2-75-billion/")

