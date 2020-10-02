#!/usr/bin/env python
# coding: utf-8

# ## The purpose of this kernel is to create features from the following great kernel and save them for further modeling:
# 
# https://www.kaggle.com/abhishek/distilbert-use-features-oof
# 
# ## If you like the kernel, consider upvoting it and the associated datasets:
# 
# https://www.kaggle.com/abhishek/transformers
# 
# https://www.kaggle.com/abhishek/sacremoses
# 
# https://www.kaggle.com/abhishek/distilbertbaseuncased

# ### Most of the code in this kernel comes directly from:
# 
# https://www.kaggle.com/abazdyrev/use-features-oof
# 
# Please consider upvoting it!

# In[ ]:


get_ipython().system('pip install ../input/sacremoses/sacremoses-master/ > /dev/null')

import os
import sys
import glob
import torch

sys.path.insert(0, "../input/transformers/transformers-master/")
import transformers
import numpy as np
import pandas as pd
import math


# In[ ]:


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


# In[ ]:


def fetch_vectors(string_list, batch_size=64):
    # inspired by https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
    DEVICE = torch.device("cuda")
    tokenizer = transformers.DistilBertTokenizer.from_pretrained("../input/distilbertbaseuncased/")
    model = transformers.DistilBertModel.from_pretrained("../input/distilbertbaseuncased/")
    model.to(DEVICE)

    fin_features = []
    for data in chunks(string_list, batch_size):
        tokenized = []
        for x in data:
            x = " ".join(x.strip().split()[:300])
            tok = tokenizer.encode(x, add_special_tokens=True)
            tokenized.append(tok[:512])

        max_len = 512
        padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized])
        attention_mask = np.where(padded != 0, 1, 0)
        input_ids = torch.tensor(padded).to(DEVICE)
        attention_mask = torch.tensor(attention_mask).to(DEVICE)

        with torch.no_grad():
            last_hidden_states = model(input_ids, attention_mask=attention_mask)

        features = last_hidden_states[0][:, 0, :].cpu().numpy()
        fin_features.append(features)

    fin_features = np.vstack(fin_features)
    return fin_features


# In[ ]:


df_train = pd.read_csv("../input/google-quest-challenge/train.csv").fillna("none")
df_test = pd.read_csv("../input/google-quest-challenge/test.csv").fillna("none")

sample = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")
target_cols = list(sample.drop("qa_id", axis=1).columns)

train_question_body_dense = fetch_vectors(df_train.question_body.values)
train_answer_dense = fetch_vectors(df_train.answer.values)

test_question_body_dense = fetch_vectors(df_test.question_body.values)
test_answer_dense = fetch_vectors(df_test.answer.values)


# In[ ]:


import os
import re
import gc
import pickle  
import random
import keras

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import keras.backend as K

from keras.models import Model
from keras.layers import Dense, Input, Dropout, Lambda
from keras.optimizers import Adam
from keras.callbacks import Callback
from scipy.stats import spearmanr, rankdata
from os.path import join as path_join
from numpy.random import seed
from urllib.parse import urlparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import MultiTaskElasticNet

seed(42)
tf.random.set_seed(42)
random.seed(42)


# In[ ]:


data_dir = '../input/google-quest-challenge/'
train = pd.read_csv(path_join(data_dir, 'train.csv'))
test = pd.read_csv(path_join(data_dir, 'test.csv'))
print(train.shape, test.shape)
train.head()


# In[ ]:


targets = [
        'question_asker_intent_understanding',
        'question_body_critical',
        'question_conversational',
        'question_expect_short_answer',
        'question_fact_seeking',
        'question_has_commonly_accepted_answer',
        'question_interestingness_others',
        'question_interestingness_self',
        'question_multi_intent',
        'question_not_really_a_question',
        'question_opinion_seeking',
        'question_type_choice',
        'question_type_compare',
        'question_type_consequence',
        'question_type_definition',
        'question_type_entity',
        'question_type_instructions',
        'question_type_procedure',
        'question_type_reason_explanation',
        'question_type_spelling',
        'question_well_written',
        'answer_helpful',
        'answer_level_of_information',
        'answer_plausible',
        'answer_relevance',
        'answer_satisfaction',
        'answer_type_instructions',
        'answer_type_procedure',
        'answer_type_reason_explanation',
        'answer_well_written'    
    ]

input_columns = ['question_title', 'question_body', 'answer']


# > # Features

# In[ ]:


find = re.compile(r"^[^.]*")

train['netloc'] = train['url'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])
test['netloc'] = test['url'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])

features = ['netloc', 'category']
merged = pd.concat([train[features], test[features]])
ohe = OneHotEncoder()
ohe.fit(merged)

features_train = ohe.transform(train[features]).toarray()
features_test = ohe.transform(test[features]).toarray()


# In[ ]:


module_url = "../input/universalsentenceencoderlarge4/"
embed = hub.load(module_url)


# In[ ]:


embeddings_train = {}
embeddings_test = {}
for text in input_columns:
    print(text)
    train_text = train[text].str.replace('?', '.').str.replace('!', '.').tolist()
    test_text = test[text].str.replace('?', '.').str.replace('!', '.').tolist()
    
    curr_train_emb = []
    curr_test_emb = []
    batch_size = 4
    ind = 0
    while ind*batch_size < len(train_text):
        curr_train_emb.append(embed(train_text[ind*batch_size: (ind + 1)*batch_size])["outputs"].numpy())
        ind += 1
        
    ind = 0
    while ind*batch_size < len(test_text):
        curr_test_emb.append(embed(test_text[ind*batch_size: (ind + 1)*batch_size])["outputs"].numpy())
        ind += 1    
        
    embeddings_train[text + '_embedding'] = np.vstack(curr_train_emb)
    embeddings_test[text + '_embedding'] = np.vstack(curr_test_emb)
    
del embed
K.clear_session()
gc.collect()


# In[ ]:


l2_dist = lambda x, y: np.power(x - y, 2).sum(axis=1)

cos_dist = lambda x, y: (x*y).sum(axis=1)

dist_features_train = np.array([
    l2_dist(embeddings_train['question_title_embedding'], embeddings_train['answer_embedding']),
    l2_dist(embeddings_train['question_body_embedding'], embeddings_train['answer_embedding']),
    l2_dist(embeddings_train['question_body_embedding'], embeddings_train['question_title_embedding']),
    cos_dist(embeddings_train['question_title_embedding'], embeddings_train['answer_embedding']),
    cos_dist(embeddings_train['question_body_embedding'], embeddings_train['answer_embedding']),
    cos_dist(embeddings_train['question_body_embedding'], embeddings_train['question_title_embedding'])
]).T

dist_features_test = np.array([
    l2_dist(embeddings_test['question_title_embedding'], embeddings_test['answer_embedding']),
    l2_dist(embeddings_test['question_body_embedding'], embeddings_test['answer_embedding']),
    l2_dist(embeddings_test['question_body_embedding'], embeddings_test['question_title_embedding']),
    cos_dist(embeddings_test['question_title_embedding'], embeddings_test['answer_embedding']),
    cos_dist(embeddings_test['question_body_embedding'], embeddings_test['answer_embedding']),
    cos_dist(embeddings_test['question_body_embedding'], embeddings_test['question_title_embedding'])
]).T

X_train = np.hstack([item for k, item in embeddings_train.items()] + [features_train, dist_features_train])
X_test = np.hstack([item for k, item in embeddings_test.items()] + [features_test, dist_features_test])
y_train = train[targets].values


# In[ ]:


X_train = np.hstack((X_train, train_question_body_dense, train_answer_dense))
X_test = np.hstack((X_test, test_question_body_dense, test_answer_dense))


# In[ ]:


np.save('X_train', X_train)
np.save('X_test', X_test)

