#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

import warnings
warnings.simplefilter('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Load data

# In[ ]:


train = pd.read_csv("../input/google-quest-challenge/train.csv", index_col='qa_id')
train.shape


# In[ ]:


test = pd.read_csv("../input/google-quest-challenge/test.csv", index_col='qa_id')
test.shape


# In[ ]:


train.head(3).T


# ## User pages

# In[ ]:


set(train['question_user_page'].unique()) & set(test['question_user_page'].unique())


# In[ ]:


set(train['answer_user_page'].unique()) & set(test['answer_user_page'].unique())


# One intersection for question user pages and no intersections for answer user pages.

# ## Users

# Let's make user id from site and user id assuming user has the same id on all stackexchange sites.
# 
# For example, for URL https://photo.stackexchange.com/users/1024 our id will be `stackexchange_1024`

# In[ ]:


import re


def make_user_map(user_pages):
    user_map = {}
    for p in user_pages:
        # get groups from URL (https://)(photo.stackexchange)(.com/users/1024)
        a = re.search('(https:\/\/)(.*)(.com|.net)', p)
        if a:
            # get second group (photo).(stackexchange) or use whole site
            b = re.search('(.*\.)(.*)', a.group(2))
            if b:
                s = b.group(2)
            else:
                s = a.group(2)
            # get user id from (https://photo.stackexchange.com/users/)(1024)
            c = re.search('(.*\/)(\d*)', p)
            if c:
                u = c.group(2)
            else:
                u = 'unknown'
            user_map[p] = s + '_' + u
    return user_map


# In[ ]:


train_question_user_pages = train['question_user_page'].unique()
train_question_user_map = make_user_map(train_question_user_pages)
train['question_user'] = train['question_user_page'].apply(lambda x: train_question_user_map[x])


# In[ ]:


train_answer_user_pages = train['answer_user_page'].unique()
train_answer_user_map = make_user_map(train_answer_user_pages)
train['answer_user'] = train['answer_user_page'].apply(lambda x: train_answer_user_map[x])


# In[ ]:


train.head(3).T


# In[ ]:


test_question_user_pages = test['question_user_page'].unique()
test_question_user_map = make_user_map(test_question_user_pages)
test['question_user'] = test['question_user_page'].apply(lambda x: test_question_user_map[x])


# In[ ]:


test_answer_user_pages = test['answer_user_page'].unique()
test_answer_user_map = make_user_map(test_answer_user_pages)
test['answer_user'] = test['answer_user_page'].apply(lambda x: test_answer_user_map[x])


# In[ ]:


set(train['question_user'].unique()) & set(test['question_user'].unique())


# In[ ]:


set(train['answer_user'].unique()) & set(test['answer_user'].unique())


# Few intersections.
