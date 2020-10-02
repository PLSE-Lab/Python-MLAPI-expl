#!/usr/bin/env python
# coding: utf-8

# Simple Brute Force Baseline notebook for first submit. Stay tuned and upvote! ;)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


sample_submission = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
train.shape, test.shape


# In[ ]:


train.sample(3)


# In[ ]:


test.sample(3)


# In[ ]:


sample_submission.sample(3)


# # Submission

# In[ ]:


def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


# In[ ]:


def make_submit(df, target_col, filename='submission.csv'):
    sub = sample_submission.copy()
    sub['selected_text'] = df[target_col]
    sub.to_csv(filename, index=False)
    print('Submission file is done!')
    return sub.head()


# In[ ]:


# baseline: test text as selected text
make_submit(test, 'text')

