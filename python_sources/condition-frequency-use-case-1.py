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

# Any results you write to the current directory are saved as output.


# In[ ]:


import nltk


# In[ ]:


from nltk.corpus import brown


# Import brown corpus
# Determine the frequency of all words (converted into lower case) occurring in different genre of brown corpus. Store the result in brown_cfd.
# Hint: Compute the condition frequency with condition being genre and event being word.
# Print the frequency of modal words ['can', 'could', 'may', 'might', 'must', 'will'], in text collections associated with genre news, religion and romance, in form a of a table.
# Hint : Make use of tabulate method associated with a conditional frequency distribution.

# In[ ]:


brown_cfd = nltk.ConditionalFreqDist([ (genre, word.lower()) for genre in brown.categories() for word in brown.words(categories=genre) ])


# In[ ]:


brown_cfd


# In[ ]:


brown_cfd.conditions()


# In[ ]:


brown_cfd.tabulate(conditions = ['news','religion','romance'],samples = ['can', 'could', 'may', 'might', 'must', 'will'])


# In[ ]:


brown_cfd.plot(conditions = ['news','religion','romance'],samples = ['can', 'could', 'may', 'might', 'must', 'will'])


# Import inaugural corpus
# For each of the inaugural address text available in the corpus, perform the following.
# Convert all words into lower case.
# Then determine the number of words starting with america or citizen.
# Hint : Compute conditional frequency distribution, where condition is the year in which the inaugural address was delivered and event is either america or citizen. Store the conditional frequency distribution in variable ac_cfd.
# 
# Print the frequency of words ['america', 'citizen'] in year [1841, 1993].
# 
# Hint: Make use of tabulate method associated with a conditional frequency distribution.

# In[ ]:


from nltk.corpus import inaugural


# In[ ]:


cfd = nltk.ConditionalFreqDist(
          (target, fileid[:4])
          for fileid in inaugural.fileids()
          for w in inaugural.words(fileid)
          for target in ['america', 'citizen']
          if w.lower().startswith(target))


# In[ ]:


cfd.tabulate(conditions=['america', 'citizen'], samples=['1841', '1993'])


# In[ ]:


cfd.plot(conditions=['america', 'citizen'], samples=['1841', '1993'])

