#!/usr/bin/env python
# coding: utf-8

# **Likelihood Ratio Test for finding all bi-gram **

# In[ ]:


import nltk
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import RegexpTokenizer
import re
import gc
from  collections import Counter
import math


# In[ ]:


file_0=open("../input/xab",mode='r',encoding='latin-1')
file_content=file_0.read()
file_0.close()
del(file_0)
gc.collect()


# In[ ]:


temp = re.sub(r'[^a-zA-Z0-9\-\s]*', r'', file_content)
temp = re.sub(r'(\-|\s+)', ' ', temp)
del(file_content)
gc.collect()


# In[ ]:


token_nltk=nltk.word_tokenize(temp)
del(temp)
word_count = Counter(token_nltk)
#print(word_count)
gc.collect()


# In[ ]:


two_gram=nltk.ngrams(token_nltk,2)
gm=Counter(two_gram)
token=gm.keys()
frequency=gm.values()


# In[ ]:


def binary(k, n, x):
    if (x == 0.0):
        x = 0.01
    if (x == 1.0): 
        x = 0.99
    return k * math.log(x) + (n-k) * math.log(1-x)


# In[ ]:


N =sum(frequency)
length_dist_bigram=len(token)
bigrams = nltk.bigrams(token_nltk)


# In[ ]:


unique_bigram=set(bigrams)


# In[ ]:


list1=[]
list2=[]
for bi in unique_bigram:
    c12 = gm.get(bi)
    w1 = bi[0]
    w2 = bi[1]
    c1 = word_count.get(w1)
    c2 = word_count.get(w2)
    #print(c12)
    #print(bi)
    p = c2/N
    p1 = c12/c1
    p2 = (c2 - c12)/(N - c1)
    ll = binary(c12, c1, p) + binary(c2-c12, N-c1, p) - binary(c12, c1, p1) - binary(c2-c12, N-c1, p2)
    ratio = -2 * ll
    #print(bi)
    #print(ratio)
    list1.append(ratio)
    list2.append(bi)
    


# In[ ]:


df=pd.DataFrame()
df["bi-gram"]=list2
df["lv"]=list1


# In[ ]:


df.head()


# In[ ]:


df.to_csv("mlr_bigram.csv")


# In[ ]:


df_10=df.nlargest(10,"lv")


# In[ ]:


df_10


# In[ ]:




