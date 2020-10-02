#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


# Install bert-as-service
get_ipython().system('pip install bert-serving-server')
get_ipython().system('pip install bert-serving-client')


# In[ ]:


# Download and unzip the pre-trained model
get_ipython().system('wget http://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip')
get_ipython().system('unzip uncased_L-12_H-768_A-12.zip')


# In[ ]:


# Start the BERT server
bert_command = 'bert-serving-start -model_dir /kaggle/working/uncased_L-12_H-768_A-12'
process = subprocess.Popen(bert_command.split(), stdout=subprocess.PIPE)


# In[ ]:


# Start the BERT client
from bert_serving.client import BertClient
bc = BertClient()


# In[ ]:


# Compute embeddings for some test sentences
embeddings = bc.encode(['Embed a single sentence', 
                        'Can it handle periods? and then more text?', 
                        'how about periods.  and <p> html stuffs? <p>'])


# The model returns 768-dimensional embeddings:

# In[ ]:


embeddings.shape


# In[ ]:


embeddings

