#!/usr/bin/env python
# coding: utf-8

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


import json 
import re
import pandas as pd


# In[ ]:


train_head = []
nrows = 500

with open("/kaggle/input/tensorflow2-question-answering//"+'simplified-nq-train.jsonl', 'rt') as f:
    for i in range(nrows):
        train_head.append(json.loads(f.readline()))

train = pd.DataFrame(train_head)


# In[ ]:


train.head()


# In[ ]:


train.annotations[0]


# In[ ]:


def answers_extractor(row):
    annot = row['annotations'][0]
    tokens = row['document_text'].split(" ")
    
    long_answer_candidates = row['long_answer_candidates']
    short_answer_candidates = annot['short_answers']
    
    if annot['yes_no_answer'] != "None":
        yes_no_answer = annot['yes_no_answer']
    
    all_long_answer_texts = []
    for ans in long_answer_candidates:
        long_start = ans['start_token']
        long_end = ans['end_token']
        all_long_answer_texts.append(tokens[long_start:long_end])
    
    long_answer_idx = annot['long_answer']['candidate_index']
    true_long_answer_text = all_long_answer_texts[long_answer_idx]
    
    short_answer_texts = []
    for ans in short_answer_candidates:
        short_start = ans['start_token']
        short_end = ans['end_token']
        short_answer_texts.append(tokens[short_start:short_end])
        
    return tokens, long_answer_idx, all_long_answer_texts, true_long_answer_text, short_answer_texts, yes_no_answer


# In[ ]:


train[['tokens', 'long_answer_idx', 'all_long_answer_texts', 'true_long_answer_text', 'short_answer_texts', "yes_no_answer"]] = train.apply(answers_extractor, axis=1, result_type="expand")


# In[ ]:


train.head()


# In[ ]:




