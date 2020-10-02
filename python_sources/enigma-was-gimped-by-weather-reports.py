#!/usr/bin/env python
# coding: utf-8

# Let us see if we can break simple ciphers with repeated appearing words and phrase! (SPOILER: cipher 1 is quite easy)

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 7.5]
import numpy as np
import pandas as pd
import string
from collections import Counter
import nltk
import os
import re


# ### Loading Data

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


level_one = train[train['difficulty']==1].copy()


# In[ ]:


level_one.shape


# ### Chipher alphabet

# In[ ]:


alp = pd.Series(Counter(''.join(level_one['ciphertext'])))
alp.head(10)


# In[ ]:


alp.shape


# We now know the number of distinct characters in the cipher text. Coupled this with interneuron's awesome analysis in the other notebook (cipher characters kind of follow a similar distribution to typical letter distribution), we can guess that cipher 1 uses a substitution algorithm.

# ### Loading Plaintext

# In[ ]:


from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset='train')


# ### Count the Most Common Opening Words
# 
# Our strategy is to find common words and phrases at the beginning or end of a text, find common patterns in the corresponding parts of the ciphertext, and try to connect them with each other.

# In[ ]:


heads = [x[:6] for x in news['data']]


# In[ ]:


Counter(heads).most_common(10)


# We see that `From: ` is the most common beginning, so it should also mean that primative encryption algorithms will map them to the same block of ciphertext at corresponding positions.

# ### Count the Most Common Opening Ciphertext Characters

# In[ ]:


level_one['ciphertext'].apply(lambda x: x[:6]).value_counts().reset_index().head(10)


# We can confidently infer that
# > 'From: ' -> '*#^-G1'

# In[ ]:


subs = {
    'F': '*',
    'r': '#',
    'o': '^',
    'm': '-',
    ':': 'G',
    ' ': '1'
}
subs = {v:k for k, v in subs.items()}


# In[ ]:


def decipher(ciphertext):
    return ''.join([subs[c] if c in subs.keys() else '?' for c in ciphertext])

def undeciphered(ciphertext):
    return ''.join(['?' if c in subs.keys() else c for c in ciphertext])


# In[ ]:


level_one['ciphertext'].head(10).apply(decipher).reset_index()


# We can now iterate this process and look for the next matching target: `Subject: `

# In[ ]:


heads = [x[:9] for x in news['data'] if x[:6] != 'From: ']
Counter(heads).most_common()


# In[ ]:


heads = level_one['ciphertext'].apply(lambda x: x[:9]).value_counts().head(20).reset_index()
heads['deciphered'] = heads['index'].apply(decipher)
heads


# Haha! Row 2 is probably 'Subject: '! We move on to the next target:

# In[ ]:


subs = {v: k for k, v in zip('From: ', '*#^-G1')}
subs.update({v: k for k, v in zip('Subject', '>cX_t')})


# In[ ]:


heads = [x[:14] for x in news['data'] if x[:6] != 'From: ' and x[:9] != 'Subject: ']
Counter(heads).most_common()


# In[ ]:


heads = level_one['ciphertext'].apply(lambda x: x[:14]).value_counts().head(20).reset_index()
heads['deciphered'] = heads['index'].apply(decipher)
heads


# Row 2 now should be 'Organization'!

# In[ ]:


subs = {v: k for k, v in zip('From: ', '*#^-G1')}
subs.update({v: k for k, v in zip('Subject', '>cX_t')})
subs.update({v: k for k, v in zip('Organization', '%#dOahOta^')})


# In[ ]:


heads = level_one['ciphertext'].apply(lambda x: x[:14]).value_counts().head(10).reset_index()
heads['deciphered'] = heads['index'].apply(decipher)
heads


# Looking at `Subject: Re: `

# In[ ]:


heads = [x[:13] for x in news['data'] if x[:6] != 'From: ']
Counter(heads).most_common(10)


# In[ ]:


heads = level_one['ciphertext'].apply(lambda x: x[:13]).value_counts().head(10).reset_index()
heads['deciphered'] = heads['index'].apply(decipher)
heads


# In[ ]:


heads['index'][2]


# In[ ]:


subs = {v: k for k, v in zip('From: ', '*#^-G1')}
subs.update({v: k for k, v in zip('Subject', '>cX_t')})
subs.update({v: k for k, v in zip('Organization', '%#dOahOta^')})
subs.update({v: k for k, v in zip('R', '\x1e')})


# In[ ]:


heads = level_one['ciphertext'].apply(lambda x: x[:13]).value_counts().head(10).reset_index()
heads['deciphered'] = heads['index'].apply(decipher)
heads


# Onto the next: `?i?tribution:`

# In[ ]:


heads = level_one['ciphertext'].apply(lambda x: x[:13]).value_counts().head(50).reset_index()
heads['deciphered'] = heads['index'].apply(decipher)
heads[heads['deciphered'].apply(lambda x: x[:4] != 'From')]


# In[ ]:


heads = [x[:13] for x in news['data'] if x[:6] != 'From: ' and x[:9] != 'Subject: ']
Counter(heads).most_common(10)


# We found `Distribution: `!

# In[ ]:


subs = {v: k for k, v in zip('From: ', '*#^-G1')}
subs.update({v: k for k, v in zip('Subject', '>cX_t')})
subs.update({v: k for k, v in zip('Organization', '%#dOahOta^')})
subs.update({v: k for k, v in zip('R', '\x1e')})
subs.update({v: k for k, v in zip('Ds', 'xv')})


# In[ ]:


heads = level_one['ciphertext'].apply(lambda x: x[:13]).value_counts().head(50).reset_index()
heads['deciphered'] = heads['index'].apply(decipher)
heads[heads['deciphered'].apply(lambda x: x[:4] != 'From')]


# We now have figured out enough letters. We might be able to directly map one text to the source.

# In[ ]:


level_one['ciphertext'].apply(decipher).iloc[3]


# In[ ]:


np.where([re.search(r'Samue\w Ross', s) != None for s in news['data']])


# In[ ]:


news['data'][1646]


# Found it! Not exact match but same post!

# In[ ]:


level_one['ciphertext'].iloc[3]


# In[ ]:


level_one['ciphertext'].apply(decipher).iloc[3]


# In[ ]:


level_one['ciphertext'].apply(undeciphered).iloc[3]


# In[ ]:


news['data'][1646][:300]


# In[ ]:


subs = {v: k for k, v in zip('From: ', '*#^-G1')}
subs.update({v: k for k, v in zip('Subject', '>cX_t')})
subs.update({v: k for k, v in zip('Organization', '%#dOahOta^')})
subs.update({v: k for k, v in zip('R', '\x1e')})
subs.update({v: k for k, v in zip('Ds', 'xv')})
subs.update({v: k for k, v in zip('s', 'v')})
subs.update({v: k for k, v in zip('6@vl.d()\nBkfhp!uywL28', '5bz8\x08A|ysJf]0\'P@oWFH,')})
subs.update({v: k for k, v in zip('N-pHMEAyTKIGCJW01', '\x7fq9geE/\x10{w:"2}l\\L')})


# This step is just to match as many letters as possible by comparing both texts. We now have the results:

# In[ ]:


level_one['ciphertext'].apply(decipher).iloc[3]


# In[ ]:


level_one['ciphertext'].apply(undeciphered).iloc[3]


# There appears to be some plain text case swap (or it could be in the source text already). But otherwise, this message is cracked!

# current cipher alphabet coverage:

# In[ ]:


len(subs.keys()) / len(alp)


# We have not cracked the full alphabet but this should be sufficient to continue working out all difficulty 1 texts.

# In[ ]:


for i in range(10):
    print(level_one['ciphertext'].apply(decipher).iloc[i])
    print('-' * 30)


# In[ ]:


for i in range(10):
    print(level_one['ciphertext'].apply(decipher).iloc[-i-1])
    print('-' * 30)


# In[ ]:




