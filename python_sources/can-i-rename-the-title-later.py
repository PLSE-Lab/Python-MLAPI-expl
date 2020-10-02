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


def contains_weird_digit(x):
    for a in x:
        if not (str.isalnum(a) or str.isspace(a) or a in '!?.,;-()\'":/\\$+=#@%&'):
            return True
    return False

train = pd.read_csv("../input/train.csv")
train = train.dropna()

train.question1 = train.question1.apply(lambda x: x.lower())
train.question2 = train.question2.apply(lambda x: x.lower())

train.loc[train.question1.apply(contains_weird_digit) , "question1"] = np.nan
train.loc[train.question2.apply(contains_weird_digit) , "question1"] = np.nan
train.dropna(inplace=True)


# In[ ]:


import itertools
words_lst_iterator1 = itertools.chain(*[''.join(map(lambda x: x if str.isalnum(x) else ' ', q)).split() for q in train.question1])
words_lst_iterator2 = itertools.chain(*[''.join(map(lambda x: x if str.isalnum(x) else ' ', q)).split() for q in train.question2])

from collections import Counter
word_counter = Counter(itertools.chain(words_lst_iterator1, words_lst_iterator2))

words = sorted(word_counter.keys(), key = lambda x: word_counter[x], reverse = True)


# In[ ]:


word_to_freq = {}


# In[ ]:


# prepares a question string for the "question_structure" feature. 
# This method returns a list of 1hot encoding of the words.
def prepare_for_question_structure(question):
    
    
    
    


# In[ ]:


import keras


# In[ ]:




