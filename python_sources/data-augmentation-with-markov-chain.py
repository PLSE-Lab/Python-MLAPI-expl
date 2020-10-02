#!/usr/bin/env python
# coding: utf-8

# Orginally want to try back-translation, but that requires internet connection which is not allowed if I want to submit. So instead, I tried the data augmentation method metioned here https://www.kaggle.com/jpmiller/extending-train-data-with-markov-chains-auc. It is using markov chain to learn the word sequence and generate new ones. However, I saw huge overfiting on my own validation set. So I didn't continue on this method, but still, it is interesting to see the machine generated sentences.

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


import markovify as mk
from joblib import Parallel, delayed


# In[ ]:


train = pd.read_csv('../input/train.csv')


# In[ ]:


insincere = train.loc[train.target==1, ['question_text', 'target']]
insincere.head()


# In[ ]:


nchar = int(insincere.question_text.str.len().median())
nchar


# In[ ]:


text_model = mk.Text(insincere['question_text'].tolist())


# In[ ]:


def data_augment():
    return text_model.make_short_sentence(nchar)


# In[ ]:


parallel = Parallel(-1, backend="threading", verbose=5)


# In[ ]:


count = 1000
aug_data = parallel(delayed(data_augment)() for _ in range(count))


# In[ ]:


aug_data[:5]


# I then tried my same model on the augemented training data, and see huge overfitting on my validation set. I guess the reason is that it does not change the vocabulary at all, and does not change the word order very much. Thus it does not provide much diversity to the data.
