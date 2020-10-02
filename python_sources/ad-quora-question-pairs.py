#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
df_test.head()


# 

# In[ ]:


#df_train.iloc[4]
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
x = df_train.ix[:,'is_duplicate']
#n, bins, patches = plt.hist(x)
#plt.show()
#pd.crosstab(index=df_train["is_duplicate"],columns="count")
df_train[df_train['is_duplicate'] == 1].head()
#df_train.iloc[0 , [2:3]]


# In[ ]:


import nltk
from nltk.book import *


# In[ ]:


text1
#A concordance view shows us every occurrence of a given word, together with some context
#text1.concordance("monstrous") 
#other words appear in a similar range of contexts
#text1.similar("monstrous")

#The term common_contexts allows us to examine just the contexts that are shared by
#two or more words
text2.common_contexts(["monstrous", "very"])
text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])


# In[ ]:


ex1 = ['The' , 'book', 'is', 'good']
ex1
sorted(ex1)
len(set(ex1))


# In[ ]:


text4[173]


# In[ ]:


text4.index('awaken')


# In[ ]:


fdist1 = FreqDist(text1)
fdist1
vocabulary1 = fdist1.keys()
fdist1.plot(70, cumulative=True)


# In[ ]:





# In[ ]:


len(text3)
#in a set, all duplicates are collapsed together
set(text3)
sorted(set(text3))
len(set(text3))

#count how often a word occurs in a text
text1.count("monstrous")

def lexical_diversity(text):
    return len(text) / len(set(text))

def percentage(count, total):
    return 100 * count / total


# In[ ]:


sent = ['word1', 'word2', 'word3', 'word4', 'word5',
'word6', 'word7', 'word8', 'word9', 'word10']
sent[:3]
saying = ['After', 'all', 'is', 'said', 'and', 'done',
'more', 'is', 'said', 'than', 'done']
saying
tokens = set(saying)
tokens = sorted(tokens)
tokens

#tokens[-2:]


# In[ ]:




