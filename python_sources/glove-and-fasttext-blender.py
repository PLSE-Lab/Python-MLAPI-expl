#!/usr/bin/env python
# coding: utf-8

# This is a simple average of two neaural networks with two word vesctor embeddings: [GLOVE](https://www.kaggle.com/tunguz/bi-gru-cnn-poolings-gpu-kernel-version) and [FastText](https://www.kaggle.com/tunguz/bi-gru-lstm-cnn-poolings-fasttext). The scripts in those two kernels are modified version of [Meng Ye's Notebook](https://www.kaggle.com/konohayui/bi-gru-cnn-poolings/code). Those scripts also output network weights, and those could potentially be used for different projects/purposes in the future. 

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


glove_submission = pd.read_csv("../input/bi-gru-cnn-poolings-gpu-kernel-version/submission.csv")
fasttext_submission = pd.read_csv("../input/bi-gru-lstm-cnn-poolings-fasttext/submission.csv")


# In[3]:


glove_submission.head()


# In[4]:


fasttext_submission.head()


# In[5]:


categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


# In[8]:


blend = fasttext_submission.copy()
blend[categories] = (0.5*fasttext_submission[categories].values +
                     0.5*glove_submission[categories].values)
blend.head()


# In[ ]:


blend.to_csv("blend.csv", index=False)


# In[ ]:




