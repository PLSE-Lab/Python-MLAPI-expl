#!/usr/bin/env python
# coding: utf-8

# Hey guys, this is just a very early version of a glance to this challenge. It will remain be a draft in such early stage. Apologize if it is difficult to read, I understand it looks messy now, but I'll modify it soon after.
# 
# Version 1: Instant ideas. 2019-03-28

# In[ ]:


import os
import warnings
import re

import numpy as np
import pandas as pd

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm, tqdm_notebook

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

colors = cm.gist_rainbow(np.linspace(0, 1, 10))

warnings.filterwarnings("ignore")

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

pd.options.display.max_rows = 16
pd.options.display.max_columns = 32


# In[ ]:


text = pd.read_csv('../input/training.csv')
ciph = pd.read_csv('../input/test.csv')


# In[ ]:


text.head()


# In[ ]:


ciph.head()


# In[ ]:


ciph.difficulty.value_counts()


# This first idea is that we can test the amount of information contained in the text. Maybe we start with the easiest in the begining.

# In[ ]:


text['word_list'] = text.text.str.split()
text['word_num'] = text['word_list'].map(len)
text.head()


# In[ ]:


ciph_4 = ciph[ciph.difficulty == 4]
ciph_4.set_index('ciphertext_id',inplace=True)
ciph_4.drop(columns = 'difficulty')


# In[ ]:


ciph_4['ciphertextlist'] = ciph_4.ciphertext.str.split()


# In[ ]:


ciph_4.head(10)


# In[ ]:


text['word_num'].value_counts()


# In[ ]:


ciph_4['length'] = ciph_4.ciphertextlist.map(len)
ciph_4.length.value_counts()


# Not surprisingly, the length of cipher is also processed to prevent leakage. Actually, there are two ways of understanding the space in difficult 4 cipher:
# 1. There are 11 types of 'alphabets' to present information, each alphabet carries some kind of information.
# 2. The 0-9 numbers contain information, the space is only a way to separate 'words' or 'letters'.
# 
# Let's try figure it out.

# In[ ]:


for i in range(10):
    ciph_char = str(i)
    ciph_4[ciph_char] = ciph_4.ciphertext.map(lambda x: len(re.findall(ciph_char,x)))
    ciph_4[ciph_char] = ciph_4[ciph_char] / ciph_4.length
ciph_4.head()    


# In[ ]:


colors[0].reshape(-1,4)


# In[ ]:


for i in range(10):
    ciph_4[str(i)].plot(color = colors[i].reshape(-1,4),legend=True)


# The figure above shows the weight of each number in the ciphertext, where the weight of space should be length-1 / length. Eventhough it's natural to believe space in cipher is used for seperation, but we still don't have enough reason to reject hypothesis 1 so far.

# In[ ]:


for i in range(10):
    sns.kdeplot(ciph_4[str(i)],color = colors[i])


# It's seems each digit follows some kind of distribution similar to normal distribution, and it should worth digging deeper. But we head on first.

# In[ ]:


ciph_4[[str(x) for x in range(10)]].std()/ciph_4[[str(x) for x in range(10)]].mean()


# In[ ]:




