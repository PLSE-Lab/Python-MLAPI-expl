#!/usr/bin/env python
# coding: utf-8

# # Conversational Question Answering Dataset (CoQA)
# ### 127,000+ questions with answers collected from 8000+ conversations
# This notebook will teach you how properly deal with json files

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


get_ipython().run_cell_magic('bash', '', 'head -c 1000 "../input/conversational-question-answering-dataset-coqa/coqa-dev-v1.0.json"')


# In[ ]:


data=json.load((open('../input/conversational-question-answering-dataset-coqa/coqa-dev-v1.0.json')))
qas=json_normalize(data['data'], ['questions'],['source','id','story'])
ans=json_normalize(data['data'], ['answers'],['id'])
train_df = pd.merge(qas,ans, left_on=['id','turn_id'],right_on=['id','turn_id'] )
train_df.loc[10:30,['turn_id','input_text_x','input_text_y','span_text'] ]


# In[ ]:


train_df['q_first_word']=train_df['input_text_x'].str.lower().str.extract(r'(\w+)')
train_df['q_first_two_words']=train_df['input_text_x'].str.lower().str.extract(r'^((?:\S+\s+){1}\S+).*')
train_df.groupby('q_first_word').count().sort_values(by='input_text_x',ascending=False).head(30)


# In[ ]:


train_df.groupby('q_first_two_words').count().sort_values(by='input_text_x',ascending=False).head(30)

