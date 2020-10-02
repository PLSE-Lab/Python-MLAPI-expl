#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import gzip
import pandas as pd
import numpy as np
from tqdm import tqdm
import nltk
import pickle
import collections
from pandas.io.json import json_normalize
import re
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'head -c 1000 "../input/commonsenseqa-nlp-dataset/train_rand_split.jsonl"')


# In[ ]:


data = []
with open('../input/commonsenseqa-nlp-dataset/train_rand_split.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))
    extracted_data = json_normalize(data)
    extracted_data.columns = extracted_data.columns.map(lambda x: x.split(".")[-1])
extracted_data


# In[ ]:


num_choices = len(extracted_data['choices'][0])

def get_choices(options, val):
    option = options[int(val)]
    return option.get('text')

choices = np.arange(num_choices)
choices = choices.astype('str')
for c in choices:
    extracted_data['choice_' + c] = extracted_data['choices'].apply(lambda x: get_choices(x, c))
answer_match = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}

def get_label(answer):
    return answer_match.get(answer)
extracted_data['label'] = extracted_data['answerKey'].apply(lambda x: get_label(x))

extracted_data

