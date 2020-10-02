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


#train = pd.read_csv('../input/train.csv').fillna(' ')
test = pd.read_csv('../input/test.csv').fillna(' ')

#class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

#train_text = train['comment_text'].values.tolist()
test_text = test['comment_text'].values.tolist()
#train_target = train[class_names].values

#np.save('train_target', train_target)


# In[ ]:


len(test_text)/80


# In[ ]:


del test
gc.collect()


# In[ ]:


start_time = time()
embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
time() - start_time


# In[ ]:


test_text = [test_text[i:i + 2000] for i in range(0, len(test_text), 2000)]


# In[ ]:


len(test_text)


# In[ ]:


embeddings_1 = []
for i in tqdm(range(40, 77)):
    embeddings = embed(test_text[i])
    embeddings_1.append(embeddings)


# In[ ]:


del test_text
gc.collect()


# In[ ]:


with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    for i in tqdm(range(37)):
        test_embeddings = session.run(embeddings_1[i])
        np.save('test_embeddings'+str(i+40)+'.npy', test_embeddings)


# In[ ]:




