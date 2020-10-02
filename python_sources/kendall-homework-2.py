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


import numpy as np
import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.probability import FreqDist
from heapq import nlargest
from collections import defaultdict


# In[ ]:


def max(l):
    maximum = 0
    place = 0
    for i in range(len(l)):
        if(l[i]>maximum):
            maximum = l[i]
            place = i
    return place


# In[ ]:


def main(articleURL):
    response = requests.get(articleURL)
    response.encoding = 'utf-8'
    data = response.text
    soup = BeautifulSoup(data)
    #print(soup)
    if(soup.find('article') == None):
        print("This is not an article!")
    else:
        text = ' '.join(map(lambda p: p.text, soup.find_all('article')))
        text.encode('ascii', 'ignore')
        sents = sent_tokenize(text)
        word_sent = word_tokenize(text.lower())
        _stopwords = set(stopwords.words('english') + list(punctuation))

        # Filter out stopword
        word_sent=[word for word in word_sent if word not in _stopwords]
        #print(word_sent)
        freq = FreqDist(word_sent)
        #print(freq)
        word_list = nlargest(10, freq, key=freq.get)
        #print(word_list)

        # We want to create a signifcant score ordered by highest frequency
        ranking = defaultdict(int)
        for i,sent in enumerate(sents):
            for w in word_tokenize(sent.lower()):
                if w in freq:
                    ranking[i] += freq[w]
        # print(ranking)
        # Top 4 Sentences
        sents_idx = nlargest(4, ranking, key=ranking.get)
        
        # We want to make the sentences human readable
        for i in range(len(sents)):
            sents[i] = sents[i].replace("\n", " ")
            #print(sents[i] +"\n\n\n")
            
        freqs = [0]*len(sents)
        freqs=list(map(int, freqs))
        #print(freqs)
        
        #Go sentence by sentence
        for i in range(len(sents)):
   #Count up each occurrence of each most occurring word
            for j in range(len(word_list)):
                freqs[i] += sents[i].count(word_list[j])
            #print(sents[i])
            #print(freqs[i])
            
            #We have two lists: all sentences, and importance of all sentences.
            #We will find the most important sentence, print it, then lower its 
            #importance to find the next most important sentence.
            
        #print(str(max(freqs)))
        for i in range(4):
            print(sents[max(freqs)])
            freqs[max(freqs)]=0
            print("\n\n\n")


# In[ ]:


main("https://www.chicagotribune.com/news/ct-xpm-1995-12-13-9512130180-story.html")


# In[ ]:


main("https://www.facebook.com/")


# In[ ]:


main("https://ben.desire2learn.com/d2l/le/content/335708/viewContent/1703635/View")


# In[ ]:


main("https://www.geekwire.com/2018/tesla-posts-historic-quarterly-report-positive-net-income-cash-flow/")


# In[ ]:


main("https://www.eurogamer.net/articles/2018-10-25-the-human-cost-of-red-dead-redemption-2")

