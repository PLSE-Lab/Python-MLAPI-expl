#!/usr/bin/env python
# coding: utf-8

# Tensorflow hub provides a very nice [universal sentence encoder module](https://tfhub.dev/google/universal-sentence-encoder/2). Unfortunately, it can only bne accessed through an online connection, which is not permitted in this competition. I am still leaving the kernel here for educational purposes.

# In[ ]:


import os
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
import gc

import lightgbm as lgb

from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression


import matplotlib.pyplot as plt
import seaborn as sns

from __future__ import absolute_import, division, print_function
import tensorflow as tf
import tensorflow_hub as hub

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[ ]:


train = pd.read_csv('../input/train.csv').fillna(' ')
test = pd.read_csv('../input/test.csv').fillna(' ')


# In[ ]:


test_text = test['question_text'].values.tolist()


# In[ ]:


start_time = time()
embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
time() - start_time


# In[ ]:


embeddings = embed(test_text)


# In[ ]:


with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    test_embeddings = session.run(embeddings)


# In[ ]:


test_embeddings.shape


# For train text we'll do somethign slightly different. In my previous attempts at emebdding large lists of texts in a Kaggle kernel I ran into mamory/timout issues. So I decided to break the list into smaller chunks, end embed one of them at the time. We'll do something like that here too.
# 
# First, let's extract the training text from the train file:

# In[ ]:


train_text = train['question_text'].values.tolist()


# We'd like to chunk it into smaller segments, each of which is approximatley the same lenght as the test text. Turns out that if we aim for about 25 chunks we can get there:

# In[ ]:


len(train_text)/25


# Now we "chunk" it:

# In[ ]:


train_text = [train_text[i:i + 52250] for i in range(0, len(train_text), 52250)]


# In[ ]:


len(train_text)


# And now we embed each chunk individually. This takes about 15 minutes. 

# In[ ]:


embeddings_train = []
for i in tqdm(range(25)):
    embeddings = embed(train_text[i])
    embeddings_train.append(embeddings)


# In[ ]:


train_embeddings_all = []
with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    for i in tqdm(range(25)):
        train_embeddings = session.run(embeddings_train[i])
        train_embeddings_all.append(train_embeddings)


# In[ ]:


del train_text, test_text
gc.collect()


# In[ ]:


train_embeddings_all = np.vstack(train_embeddings_all)


# In[ ]:


train_embeddings_all.shape


# In[ ]:


train_target = train['target'].values
del train, test
gc.collect()


# In[ ]:


kf = KFold(n_splits=5, shuffle=True, random_state=43)
test_pred_tf = 0
oof_pred_tf = np.zeros([train_embeddings_all.shape[0],])

for i, (train_index, val_index) in tqdm(enumerate(kf.split(train_embeddings_all))):
    x_train, x_val = train_embeddings_all[train_index,:], train_embeddings_all[val_index,:]
    y_train, y_val = train_target[train_index], train_target[val_index]
    classifier = LogisticRegression(class_weight = "balanced", C=0.5, solver='sag')
    classifier.fit(x_train, y_train)
    val_preds = classifier.predict_proba(x_val)[:,1]
    preds = classifier.predict_proba(test_embeddings)[:,1]
    test_pred_tf += 0.2*preds
    oof_pred_tf[val_index] = val_preds


# In[ ]:


np.save('train_embeddings_all', train_embeddings_all)
np.save('test_embeddings', test_embeddings)
np.save('train_target', train_target)


# In[ ]:


pred_train = (oof_pred_tf > 0.8).astype(np.int)
f1_score(train_target, pred_train)


# In[ ]:


test = pd.read_csv('../input/test.csv').fillna(' ')
pred_test = (test_pred_tf> 0.8).astype(np.int)
submission = pd.DataFrame.from_dict({'qid': test['qid']})
submission['prediction'] = pred_test
submission.to_csv('submission.csv', index=False)


# In[ ]:




