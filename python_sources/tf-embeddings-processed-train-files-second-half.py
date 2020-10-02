#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from time import time

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from __future__ import absolute_import, division, print_function
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import gc

import os
from tqdm import tqdm
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/train.csv').fillna(' ')
#test = pd.read_csv('../input/test.csv').fillna(' ')

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train_text = train['comment_text'].values.tolist()
#test_text = test['comment_text'].values.tolist()
train_target = train[class_names].values

np.save('train_target', train_target)

del train_target
gc.collect()


# In[ ]:


len(train_text)/80


# In[ ]:


del train
gc.collect()


# In[ ]:


start_time = time()
embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
time() - start_time


# In[ ]:


train_text = [train_text[i:i + 2000] for i in range(0, len(train_text), 2000)]


# In[ ]:


len(train_text)


# In[ ]:


embeddings_1 = []
for i in tqdm(range(40, 80)):
    embeddings = embed(train_text[i])
    embeddings_1.append(embeddings)


# In[ ]:


del train_text
gc.collect()


# In[ ]:


with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    for i in tqdm(range(40)):
        train_embeddings = session.run(embeddings_1[i])
        np.save('train_embeddings'+str(i+40)+'.npy', train_embeddings)

