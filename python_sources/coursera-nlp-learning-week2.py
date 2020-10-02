#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# # NLTK=Natural Language Toolkit. Part 1
# 
# * Counting words, counting frequencies of words
# * Finding sentence boundaries

# In[ ]:


import nltk


# In[ ]:


from nltk.book import *


# In[ ]:


sents() # shows one sentence from each corpora


# In[ ]:


text7


# In[ ]:


sent7 # parsed out already


# In[ ]:


len(sent7)


# In[ ]:


len(text7) # how many words in total


# In[ ]:


len(set(text7)) # to see the number of unique words


# In[ ]:


list(set(text7))[:10] # to see first 10 unique words


# # Word Frequency

# In[ ]:


# frequency of words
dist = FreqDist(text7) # FreqDist is a part of nltk library
len(dist) # returns the same number of unique words


# In[ ]:


vocab1 = list(dist.keys()) # to get the actual words
vocab1[:10]


# In[ ]:


dist['Vinken']


# In[ ]:


freqwords = [w for w in vocab1 if len(w)>5 and dist[w]>100]
freqwords


# # Normalization and Stemming
# 
# * the same word may occur in different forms

# In[ ]:


input1 = 'List listed lists listing listings'


# In[ ]:


words1 = input1.lower().split(" ") # don't want to distinguish "List" with "list"
words1 # normalization is done


# In[ ]:


# stemming is coming
porter = nltk.PorterStemmer() # to find to root form of any given word


# In[ ]:


[porter.stem(t) for t in words1]


# # Variation of stemming: Lemmatization

# In[ ]:


udhr = nltk.corpus.udhr.words('English-Latin1')
udhr[:20]


# In[ ]:


[porter.stem(t) for t in udhr[:20]]
# resulting list: some words are not valid words


# In[ ]:


# lemmatization does stemming but all resulting words are valid
WNlemma = nltk.WordNetLemmatizer()


# In[ ]:


[WNlemma.lemmatize(t) for t in udhr[:20]] # all words are valid


# # Tokenization, again

# In[ ]:


text11 = "Children shouldn't drink a sugary drink before bed."


# In[ ]:


text11.split(" ") # it's keeping the full stop with the word


# In[ ]:


nltk.word_tokenize(text11)


# In[ ]:


#nltk has a sentense splitter also
text12 = "This is the first sentence. A gallon of milk in the U.S. costs $2.99. Is this the third sentence? Yes, it is!"
sentences = nltk.sent_tokenize(text12)
sentences


# # NLTK. Part 2
# * part of speech tagging

# In[ ]:


nltk.help.upenn_tagset('MD')


# In[ ]:


#apply nltk's word tokenizer
text13 = nltk.word_tokenize(text11)
text13


# In[ ]:


nltk.pos_tag(text13)


# ### Ambiguity in POS tagging:

# In[ ]:


text14 = "Visiting aunts can be a nuisance"
text15 = nltk.word_tokenize(text14)
nltk.pos_tag(text15)


# In[ ]:


text15 = nltk.word_tokenize("Alice loves Bob")
grammar = nltk.CFG.fromstring("""
S -> NP VP
VP -> V NP
NP -> 'Alice' | 'Bob'
V -> 'loves'
""")
parser = nltk.ChartParser(grammar)
trees = parser.parse_all(text15)
for tree in trees:
    print(tree)


# In[ ]:


text16 = nltk.word_tokenize("I saw the man with a telescope")
grammar1 = nltk.CFG.fromstring("""
S -> NP VP
VP -> V NP | VP PP
PP -> P NP
NP -> DT N | DT N PP | 'I'
DT -> 'a' | 'the'
N -> 'man' | 'telescope'
V -> 'saw'
P -> 'with'
""")
parser1 = nltk.ChartParser(grammar1)
trees1 = parser1.parse_all(text16)
for tree in trees1:
    print(tree)


# # NLTK and Parse Tree Collection

# In[ ]:


from nltk.corpus import treebank


# In[ ]:


text17 = treebank.parsed_sents('wsj_0001.mrg')[0]
print(text17)


# In[ ]:




