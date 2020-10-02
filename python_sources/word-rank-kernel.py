#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import pickle
import gensim
# Any results you write to the current directory are saved as output.


# In[ ]:


# # Create and dump
# spell_model = gensim.models.KeyedVectors.load_word2vec_format('../input/wikinews300d1mvec/wiki-news-300d-1M.vec')
# words = spell_model.index2word
# w_rank = {}
# for i,word in enumerate(words):
#     w_rank[word] = i
    
# with open('word-rank-for-spell-check.pkl', 'wb') as f:
#     pickle.dump(w_rank, f)


# In[ ]:


# Load 
with open('../input/word-rand-for-spell-check/word-rank-for-spell-check.pkl', 'rb') as f:
    word_rank = pickle.load(f)


# In[ ]:


WORDS = word_rank
# https://www.kaggle.com/cpmpml/spell-checker-using-word2vec
def words(text): return re.findall(r'\w+', text.lower())
def P(word): 
    "Probability of `word`."
    # use inverse of rank as proxy
    # returns 0 if the word isn't in the dictionary
    return - WORDS.get(word, 0)
def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)
def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or [word])
def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)
def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)
def singlify(word):
    return "".join([letter for i,letter in enumerate(word) if i == 0 or letter != word[i-1]])


# In[ ]:


# Test frequency
print(WORDS['to'])
print(WORDS['0'])
print(WORDS['print'])
print(WORDS['HELLO'])
print(WORDS['shakespear'])

