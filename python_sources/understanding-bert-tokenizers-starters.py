#!/usr/bin/env python
# coding: utf-8

# # **BERT Base Uncased Exploration**

# > Hello All, Hope all doing good.
# > 
# > The goal of this notebook is to understand how BERT tokenizers look like and little deep dive into that. 
# > 
# > This notebook was created for my learning and sharing the same. I will keep updating this notebook to understand different tokenizers.
# > 
# > Thanks to ChrisMc and many more which led me to understand BERT more closer. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import torch
from pytorch_pretrained_bert import BertTokenizer

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# # **BERT Base uncased - Deep dive** # 
# 
# 
# 1. This vocab has around 30k tokens with around 768 features for each tokens
# 2. Below is how vocab looks like
# 
#    1. [PAD]
#    2. [unused0] - [unused98]
#    3. [UNK]
#    4. [CLS]
#    5. [SEP]
#    6. [MASK]
#    7. [unused99] - [unused993]
#    8. Single characters (from '!' till '~')
#    9. Whole words and subwords (starts from 'the', Reason the comes is by frequency of occurance while BERT Training)
# 

# In[ ]:


with open("vocab.txt", 'w') as f:
    for token in tokenizer.vocab.keys():
        print (token)


# # **Single Character Tokens**# 
# 
# 1. We can notice it has Numbers, Alphabets, special characters, roman letters, etc.
# 2. We can see it also covered languages English, Tamil, hindi, Chinese
# 3. We have around 997 tokens covers Single letters

# In[ ]:


single_chars = []
for token in tokenizer.vocab.keys():
    if len(token) == 1:
        single_chars.append(token)
    
print (single_chars)


# In[ ]:


print ("Length of single_chars:", len(single_chars))


# # **Single Character Tokens with # **# 
# 
# This is as same as previous Single char what we saw. all characters will be prefixed with # to help tokenize subwords. 
# 
# 1. We can notice it has Numbers, Alphabets, special characters, roman letters, etc.
# 2. We can see it also covered languages English, Tamil, hindi, Chinese
# 3. We have around 997 tokens covers Single letters

# In[ ]:


single_chars_with_hash = []
for token in tokenizer.vocab.keys():
    if len(token) == 3 and token[0:2] == '##':
        single_chars_with_hash.append(token)
    
print (single_chars_with_hash)


# In[ ]:


print ("Length of single_chars_with_hash:", len(single_chars_with_hash))


# # **Words in Corpus**# 
# 
# BERT uncased have around 23209 words and rest goes into subwords
# 
# We can see it contains years, states, countries, etc
# 

# In[ ]:


words_in_corpus= []
for token in tokenizer.vocab.keys():
    if len(token) > 2 and token[0:2] != '##':
        words_in_corpus.append(token)
    
print (words_in_corpus)


# In[ ]:


print ("How many words in the corpus:", len(words_in_corpus))


# # **Subwords **# 
# 
# BERT typically break the words into different subwords for more granularity. e.g. if we see the word doing, it will break into do + ing. 
# 
# We can see the tokens and its weights given for the "do" and "ing" hash prefixed. 
# 
# Below example, we can see few subwords as "##ion", "##fully", etc. We can also see some numbers also been treated as subwords
# 
# We have around 5828 subwords in this corpus. 

# In[ ]:


words_in_corpus_with_subwords= []
for token in tokenizer.vocab.keys():
    if len(token) > 2 and token[0:2] == '##':
        words_in_corpus_with_subwords.append(token)
    
print (words_in_corpus_with_subwords)


# In[ ]:


print ("How many sub words in the corpus:", len(words_in_corpus_with_subwords))


# # **Corpus Analyis**# 
# 
# Looks like we have the longest word is of length 18 and on an average length being 11-14

# In[ ]:


lentgh_of_words_in_corpus= []
for token in tokenizer.vocab.keys():
    if len(token) > 2 and token[0:2] != '##':
        lentgh_of_words_in_corpus.append(len(token))
    
print (lentgh_of_words_in_corpus)


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(lentgh_of_words_in_corpus)


# In[ ]:




