#!/usr/bin/env python
# coding: utf-8

# # XLM-Roberta Tokenizer [Jigsaw Toxic Comment]

# I split training and tokenization into two seperate notebooks and later chained them together. This conserved TPU time and it also left more resources for training.

# In[ ]:


import os
import re
import numpy as np
import pandas as pd
import transformers
import tensorflow as tf
import tqdm.notebook as tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam


# # Helper Functions
# Through experiments, I found that cleaning the data (getting rid of usernames, ip addresses, removing symbols) does not improve model score. In some cases, it even diminished it.

# In[ ]:


def encode(texts, tokenizer, maxlen=512):
    enc_di = tokenizer.batch_encode_plus(
        texts,
        return_attention_masks=False,
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=maxlen
    )
    return np.array(enc_di['input_ids'])


# ## Loading Data

# In[ ]:


print(os.listdir('/kaggle/input/jigsaw-multilingual-toxic-comment-classification'))


# In[ ]:


eng = '/kaggle/input/jigsaw-multilingual-toxic-comment-classification/'
non_eng = '/kaggle/input/jigsaw-train-multilingual-coments-google-api/'

es = pd.read_csv(f'{non_eng}jigsaw-toxic-comment-train-google-es-cleaned.csv')
fr = pd.read_csv(f'{non_eng}jigsaw-toxic-comment-train-google-fr-cleaned.csv')
it = pd.read_csv(f'{non_eng}jigsaw-toxic-comment-train-google-it-cleaned.csv')
pt = pd.read_csv(f'{non_eng}jigsaw-toxic-comment-train-google-pt-cleaned.csv')
ru = pd.read_csv(f'{non_eng}jigsaw-toxic-comment-train-google-ru-cleaned.csv')
tr = pd.read_csv(f'{non_eng}jigsaw-toxic-comment-train-google-tr-cleaned.csv')

for df in [es, fr, it, pt, ru, tr]:
    cols = list(df.columns)[3:]
    df.toxic = df[cols].sum(axis=1)
    df.toxic.apply(lambda x: 1 if x >= 1 else 0)

eg2 = pd.read_csv(f'{eng}jigsaw-unintended-bias-train.csv')
eg2['toxic'] = eg2.toxic.round().astype(int)

valid = pd.read_csv(f'{eng}validation.csv')


# In[ ]:


bias_augment = '/kaggle/input/translated-train-bias-all-langs/All languages'
bias_es = pd.read_csv(f'{bias_augment}/train-bias-toxic-google-api-es-cleaned.csv')
bias_fr = pd.read_csv(f'{bias_augment}/train-bias-toxic-google-api-fr-cleaned.csv')
bias_it = pd.read_csv(f'{bias_augment}/train-bias-toxic-google-api-it-cleaned.csv')
bias_pt = pd.read_csv(f'{bias_augment}/train-bias-toxic-google-api-pt-cleaned.csv')
bias_ru = pd.read_csv(f'{bias_augment}/train-bias-toxic-google-api-ru-cleaned.csv')
bias_tr = pd.read_csv(f'{bias_augment}/train-bias-toxic-google-api-tr-cleaned.csv')


# ### External Data

# In[ ]:


test_ex = '/kaggle/input/toxic-comment-detection-multilingual-extended/english/english/'
test1 = pd.read_csv(f'{eng}test.csv')
testen2 = pd.read_csv(f'{test_ex}test_en.csv')
testen3 = pd.read_csv(f'{test_ex}jigsaw_miltilingual_test_translated.csv')


# In[ ]:


external = '/kaggle/input/toxic-comment-detection-multilingual-extended/archive/'
e_ru = pd.read_csv(f'{external}russian/labeled.csv')

e_tr = pd.read_csv(f'{external}turkish/troff-v1.0.tsv', sep='\t', header=0)
e_tr.label = e_tr.label.apply(lambda x: 1 if x not in ['non', 'prof'] else 0)

e_it = pd.concat([
    pd.read_csv(f'{external}italian/haspeede_FB-train.tsv', sep='\t', header=0),
    pd.read_csv(f'{external}italian/haspeede_TW-train.tsv', sep='\t', header=0)
])


# In[ ]:


e_ru.rename(columns={'comment':'comment_text'}, inplace=True)
e_tr.rename(columns={'text':'comment_text', 'label':'toxic'}, inplace=True)
e_it.rename(columns={'comment':'comment_text'}, inplace=True)


# In[ ]:


train = pd.concat([
    es[['comment_text', 'toxic']].query('toxic==0').sample(n=20000, random_state=0),
    es[['comment_text', 'toxic']].query('toxic==1'),
    fr[['comment_text', 'toxic']].query('toxic==0').sample(n=20000, random_state=0),
    fr[['comment_text', 'toxic']].query('toxic==1'),
    it[['comment_text', 'toxic']].query('toxic==0').sample(n=20000, random_state=0),
    it[['comment_text', 'toxic']].query('toxic==1'),
    pt[['comment_text', 'toxic']].query('toxic==0').sample(n=20000, random_state=0),
    pt[['comment_text', 'toxic']].query('toxic==1'),
    ru[['comment_text', 'toxic']].query('toxic==0').sample(n=20000, random_state=0),
    ru[['comment_text', 'toxic']].query('toxic==1'),
    tr[['comment_text', 'toxic']].query('toxic==0').sample(n=20000, random_state=0),
    tr[['comment_text', 'toxic']].query('toxic==1'),
    eg2[['comment_text', 'toxic']].query('toxic==1'),
    eg2[['comment_text', 'toxic']].query('toxic==0').sample(n=200000, random_state=0),
    bias_es[['comment_text', 'toxic']].query('toxic==1'),
    bias_fr[['comment_text', 'toxic']].query('toxic==1'),
    bias_it[['comment_text', 'toxic']].query('toxic==1'),
    bias_pt[['comment_text', 'toxic']].query('toxic==1'),
    bias_ru[['comment_text', 'toxic']].query('toxic==1'),
    bias_tr[['comment_text', 'toxic']].query('toxic==1')
])


# In[ ]:


train.toxic = train.toxic.round().astype(int)


# # Data Visualizations

# In[ ]:


fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.set_title(f'Count of Toxic and Non-Toxic Training Datapoints: {len(train)}')
c_toxic, c_nontoxic = len(train[train['toxic']==1]), len(train[train['toxic']==0])
labels = [f'Toxic: {c_toxic}', f'Non-Toxic {c_nontoxic}']
values = [c_toxic, c_nontoxic]
ax.bar(labels, values)
plt.show()


# # Tokenization
# Using XLM-Roberta (Large)

# In[ ]:


tokenizer = transformers.AutoTokenizer.from_pretrained('jplu/tf-xlm-roberta-large')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nx_train = encode(train.comment_text.values, tokenizer, maxlen=192)\nx_valid = encode(valid.comment_text.values, tokenizer, maxlen=192)\nx_test1 = encode(test1.content.values, tokenizer, maxlen=192)\nx_test2 = encode(testen2.content_en, tokenizer, maxlen=192)\nx_test3 = encode(testen3.translated, tokenizer, maxlen=192)')


# In[ ]:


np.save('x_train', x_train)
np.save('x_valid', x_valid)
np.save('x_test1', x_test1)
np.save('x_test2', x_test2)
np.save('x_test3', x_test3)


# In[ ]:


np.save('y_train', train.toxic.values)
np.save('y_valid', valid.toxic.values)

