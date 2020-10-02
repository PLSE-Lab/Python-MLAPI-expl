#!/usr/bin/env python
# coding: utf-8

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


# # Cut Chinese sentences with Jieba

# In[ ]:


get_ipython().system('pip install jieba')


# In[ ]:


import jieba
import re

def remove_emoji(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)


# In[ ]:


with open("train_tcn_cut.txt", "w") as text_file:
    for line in open(r'/kaggle/input/shopee-product-title-translation-open/train_tcn.csv', encoding="utf-8"):
        product_title = line.rsplit(',', 1)[0]
        sentence = remove_emoji(product_title).replace('\\n',' ')
        sentence = re.sub('[!@#$]', '', sentence)

        words = jieba.cut(sentence, cut_all=False)
        print(' '.join(x for x in words), file=text_file)


# # Build monolingual with fastText

# In[ ]:


get_ipython().system('pip install fasttext')


# In[ ]:


import fasttext
dim = 300


# ## for Chinese

# In[ ]:


model = fasttext.train_unsupervised('/kaggle/working/train_tcn_cut.txt',dim=dim)


# In[ ]:


model.words[:20]


# In[ ]:


with open("shopee_tcn.vec", "w") as text_file:
    print(f'{len(model.words)} {dim}', file=text_file)
    for w in model.words:
        stri = model.get_word_vector(w)
        vec_str = ' '.join(str(x) for x in stri)
        print(f"{w} {vec_str}", file=text_file)


# ## for English

# In[ ]:


with open("train_en_cut.txt", "w") as text_file:
    for line in open(r'/kaggle/input/shopee-product-title-translation-open/train_en.csv', encoding="utf-8"):
        product_title = line.rsplit(',', 1)[0]
        sentence = remove_emoji(product_title).replace('\\n',' ')
        print(sentence, file=text_file)


# In[ ]:


model = fasttext.train_unsupervised('/kaggle/working/train_en_cut.txt',dim=dim)


# In[ ]:


model.words[:20]


# In[ ]:


model.get_word_vector(model.words[0]).shape


# In[ ]:


with open("shopee_eng.vec", "w") as text_file:
    print(f'{len(model.words)} {dim}', file=text_file)
    for w in model.words:
        stri = model.get_word_vector(w)
        vec_str = ' '.join(str(x) for x in stri)
        print(f"{w} {vec_str}", file=text_file)
    


# In[ ]:




