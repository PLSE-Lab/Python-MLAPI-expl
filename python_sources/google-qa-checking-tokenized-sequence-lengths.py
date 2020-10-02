#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('ls -lah ../input')


# In[ ]:


get_ipython().system('pip install ../input/sacremoses/sacremoses-master/')
get_ipython().system('pip install ../input/transformers/transformers-master/')


# In[ ]:


import re
import gc
import pickle
import numpy as np
import pandas as ps
from tqdm.auto import tqdm
from pathlib import Path
from itertools import chain
from collections import Counter
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Set
from transformers import BertTokenizer

import numpy as np
import pandas as ps

import matplotlib.pyplot as plt
import seaborn as sbn

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data_dir = Path('..') / 'input' / 'google-quest-challenge'
input_dir = Path('..') / 'input'

tokenizer = BertTokenizer.from_pretrained("../input/pretrained-bert-including-scripts/uncased_L-12_H-768_A-12/uncased_L-12_H-768_A-12")

train_df = ps.read_csv(data_dir / 'train.csv')
test_df = ps.read_csv(data_dir / 'test.csv')


# In[ ]:


def combined_len(title, body, answer):
    return len(["[CLS]"] + tokenizer.tokenize(title + "," + body) + ["[SEP]"] + tokenizer.tokenize(answer) + ["[SEP]"])

def title_body_len(title, body):
    return len(tokenizer.tokenize(title + "," + body))
    
def field_len(feature):
    return len(tokenizer.tokenize(feature))


for df in (train_df, test_df):
    df["sequences_len"] = df.apply(lambda row: combined_len(row["question_title"], row["question_body"], row["answer"]), axis=1)
    df["title_body_len"] = df.apply(lambda row: title_body_len(row["question_title"], row["question_body"]), axis=1)
    df["title_len"] = df["question_title"].apply(field_len)
    df["body_len"] = df["question_body"].apply(field_len)
    df["answer_len"] = df["answer"].apply(field_len)


# In[ ]:


def mean_std(tr_vals, ts_vals):
    items = list(tr_vals) + list(ts_vals)
    return np.mean(items), np.std(items)


# In[ ]:


mean_std(train_df["title_body_len"].values, test_df["title_body_len"].values), mean_std(train_df["answer_len"].values, test_df["answer_len"].values)


# In[ ]:





# In[ ]:


for df in (train_df, test_df):
    df["title_str_len"] = df["question_title"].str.len()
    df["body_str_len"] = df["question_body"].str.len()
    df["answer_str_len"] = df["answer"].str.len()


# In[ ]:


mean_std(train_df["title_str_len"].values, test_df["title_str_len"].values), mean_std(train_df["body_str_len"].values, test_df["body_str_len"].values), mean_std(train_df["answer_str_len"].values, test_df["answer_str_len"].values) 


# In[ ]:


def num_alpha(s):
    return sum(1 for c in s if c.isalpha())

def num_nums(s):
    return sum(1 for c in s if c.isnumeric())

def low_num(s):
    return sum(1 for c in s if c.islower())

def upper_num(s):
    return sum(1 for c in s if c.isupper())

def spaces_num(s):
    return sum(1 for c in s if c.isspace())

def num_words(s):
    return len(s.split())


for df in (train_df, test_df):
    df["title_str_alpha_num"] = df["question_title"].apply(num_alpha)
    df["title_str_nums_num"] = df["question_title"].apply(num_nums)
    df["title_str_lows_num"] = df["question_title"].apply(low_num)
    df["title_str_ups_num"] = df["question_title"].apply(upper_num)
    df["title_str_spaces_num"] = df["question_title"].apply(spaces_num)
    df["title_str_words_num"] = df["question_title"].apply(num_words)
    
    df["body_str_alpha_num"] = df["question_body"].apply(num_alpha)
    df["body_str_nums_num"] = df["question_body"].apply(num_nums)
    df["body_str_lows_num"] = df["question_body"].apply(low_num)
    df["body_str_ups_num"] = df["question_body"].apply(upper_num)
    df["body_str_spaces_num"] = df["question_body"].apply(spaces_num)
    df["body_str_words_num"] = df["question_body"].apply(num_words)
    
    df["answer_str_alpha_num"] = df["answer"].apply(num_alpha)
    df["answer_str_nums_num"] = df["answer"].apply(num_nums)
    df["answer_str_lows_num"] = df["answer"].apply(low_num)
    df["answer_str_ups_num"] = df["answer"].apply(upper_num)
    df["answer_str_spaces_num"] = df["answer"].apply(spaces_num)
    df["answer_str_words_num"] = df["answer"].apply(num_words)


# In[ ]:


mean_std(train_df["title_str_alpha_num"].values, test_df["title_str_alpha_num"].values), mean_std(train_df["title_str_nums_num"].values, test_df["title_str_nums_num"].values), mean_std(train_df["title_str_lows_num"].values, test_df["title_str_lows_num"].values), mean_std(train_df["title_str_ups_num"].values, test_df["title_str_ups_num"].values), mean_std(train_df["title_str_spaces_num"].values, test_df["title_str_spaces_num"].values), mean_std(train_df["title_str_words_num"].values, test_df["title_str_words_num"].values)


# In[ ]:


mean_std(train_df["body_str_alpha_num"].values, test_df["body_str_alpha_num"].values), mean_std(train_df["body_str_nums_num"].values, test_df["body_str_nums_num"].values), mean_std(train_df["body_str_lows_num"].values, test_df["body_str_lows_num"].values), mean_std(train_df["body_str_ups_num"].values, test_df["body_str_ups_num"].values), mean_std(train_df["body_str_spaces_num"].values, test_df["body_str_spaces_num"].values), mean_std(train_df["body_str_words_num"].values, test_df["body_str_words_num"].values)


# In[ ]:


mean_std(train_df["answer_str_alpha_num"].values, test_df["answer_str_alpha_num"].values), mean_std(train_df["answer_str_nums_num"].values, test_df["answer_str_nums_num"].values), mean_std(train_df["answer_str_lows_num"].values, test_df["answer_str_lows_num"].values), mean_std(train_df["answer_str_ups_num"].values, test_df["answer_str_ups_num"].values), mean_std(train_df["answer_str_spaces_num"].values, test_df["answer_str_spaces_num"].values), mean_std(train_df["answer_str_words_num"].values, test_df["answer_str_words_num"].values)


# ## Combined sequence length distributions

# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7))

sbn.distplot(train_df["sequences_len"].values, norm_hist=False, ax=ax1)
ax1.set_xlabel("sequence length")
ax1.set_title("Num records with len > 512: " + str(np.sum(train_df["sequences_len"].values > 512)) + f", total - {train_df.shape[0]}");

sbn.distplot(test_df["sequences_len"].values, norm_hist=False, ax=ax2)
ax2.set_xlabel("sequence length")
ax2.set_title("Num records with len > 512: " + str(np.sum(test_df["sequences_len"].values > 512)) + f", total - {test_df.shape[0]}");


# ## Distribution of `question_title` lengths

# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7))

sbn.distplot(train_df["title_len"].values, norm_hist=False, ax=ax1)
ax1.set_xlabel("sequence length")
ax1.set_title("Num records with len > 512: " + str(np.sum(train_df["title_len"].values > 512)) + f", total - {train_df.shape[0]}");

sbn.distplot(test_df["title_len"].values, norm_hist=False, ax=ax2)
ax2.set_xlabel("sequence length")
ax2.set_title("Num records with len > 512: " + str(np.sum(test_df["title_len"].values > 512)) + f", total - {test_df.shape[0]}");


# ## Distribution of `body_len` lengths

# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7))

sbn.distplot(train_df["body_len"].values, norm_hist=False, ax=ax1)
ax1.set_xlabel("sequence length")
ax1.set_title("Num records with len > 512: " + str(np.sum(train_df["body_len"].values > 512)) + f", total - {train_df.shape[0]}");

sbn.distplot(test_df["body_len"].values, norm_hist=False, ax=ax2)
ax2.set_xlabel("sequence length")
ax2.set_title("Num records with len > 512: " + str(np.sum(test_df["body_len"].values > 512)) + f", total - {test_df.shape[0]}");


# ## Distribution of `answer` lengths

# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7))

sbn.distplot(train_df["answer_len"].values, norm_hist=False, ax=ax1)
ax1.set_xlabel("sequence length")
ax1.set_title("Num records with len > 512: " + str(np.sum(train_df["answer_len"].values > 512)) + f", total - {train_df.shape[0]}");

sbn.distplot(test_df["answer_len"].values, norm_hist=False, ax=ax2)
ax2.set_xlabel("sequence length")
ax2.set_title("Num records with len > 512: " + str(np.sum(test_df["answer_len"].values > 512)) + f", total - {test_df.shape[0]}");


# In[ ]:




