#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import sys
from joblib import Parallel, delayed


sys.path.insert(0, '../input/pretrained-bert-including-scripts/master/bert-master')
BERT_PRETRAINED_DIR = '../input/pretrained-bert-including-scripts/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12'


# In[ ]:


import tokenization
dict_path = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')
tokenizer = tokenization.FullTokenizer(vocab_file=dict_path, do_lower_case=True)


# In[ ]:


train_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv', nrows=10000)


# # old version

# In[ ]:


def convert_lines(example, max_seq_length,tokenizer):
    max_seq_length -=2
    all_tokens = []
    for i in range(example.shape[0]):
        tokens_a = tokenizer.tokenize(example[i])
        if len(tokens_a)>max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    return np.array(all_tokens)


# In[ ]:


get_ipython().run_cell_magic('time', '', "train_lines, train_labels = train_df['comment_text'].values, train_df.target.values \ntoken_input = convert_lines(train_lines, 25, tokenizer)")


# # new version

# In[ ]:


def convert_line(tl, max_seq_length,tokenizer):
    example = str(tl[0])
    y = tl[1]
    max_seq_length -=2
    tokens_a = tokenizer.tokenize(example)
    if len(tokens_a)>max_seq_length:
      tokens_a = tokens_a[:max_seq_length]
    one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
    return one_token, y


# In[ ]:


train_lines = zip(train_df['comment_text'].values.tolist(), train_df.target.values.tolist())


# In[ ]:


get_ipython().run_line_magic('time', '')
res = Parallel(n_jobs=4, backend='multiprocessing')(delayed(convert_line)(i, 25, tokenizer) for i in train_lines)


# ### converting to token inputs and labels is left as an exercise to the reader

# In[ ]:




