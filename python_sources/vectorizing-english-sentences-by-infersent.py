#!/usr/bin/env python
# coding: utf-8

# # Some Remarks:
# 1. The encoder only works for English sentences, so you'll need an alternative for other languages in the text
# 2. This is merely a work in progress, but this is something to refer to when you're trying to install an external package without using the internet
# 3. I will probably do some text cleaning in further versions

# In[ ]:


get_ipython().system('export NLTK_DATA=../input/infersent/punkt/punkt')
get_ipython().system('ls /kaggle/input/infersentrepo/repository/facebookresearch-InferSent-940c003')
get_ipython().system('cd /kaggle/working')
import sys
sys.path.append('/kaggle/input/infersentrepo/repository/facebookresearch-InferSent-940c003')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("/kaggle/input/infersentrepo/repository/facebookresearch-InferSent-940c003"))
import nltk
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
from random import randint
import numpy as np
import torch
from models import InferSent
get_ipython().system('ls /kaggle/input')


# In[ ]:


model_version = 1
MODEL_PATH = "/kaggle/input/infersent/infersent%s.pkl" % model_version
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))


# In[ ]:


use_cuda = False
model = model.cuda() if use_cuda else model
# If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
W2V_PATH = '/kaggle/input/quora-insincere-questions-classification/embeddings/glove.840B.300d/glove.840B.300d.txt' if model_version == 1 else ''
model.set_w2v_path(W2V_PATH)


# In[ ]:


# Load embeddings of K most frequent words
model.build_vocab_k_words(K=100000)


# In[ ]:


a=pd.read_csv('/kaggle/input/petfinder-adoption-prediction/train/train.csv')
b=a['Description'].values
f = open("/kaggle/working/pet.txt", "w")
f = open("/kaggle/working/pet.txt", "a+")


# In[ ]:


for i in range(10000):
    f.write(str(b[i])+'\n')


# In[ ]:


# Load some sentences
sentences = []
with open('/kaggle/working/pet.txt') as f:
    for line in f:
        sentences.append(line.strip())
print(len(sentences))


# In[ ]:


sentences[:5]


# In[ ]:


embeddings = model.encode(sentences, bsize=128, tokenize=False, verbose=True)
print('nb sentences encoded : {0}'.format(len(embeddings)))


# In[ ]:


len(model.encode(['the cat eats.'])[0])


# 
