#!/usr/bin/env python
# coding: utf-8

# Use Regularized version of Bi LSTM model called AWD LSTM to classify the text documents, before Language models became SOTA, BiLstm were used extensively for Text classification Tasks.
# 
# we have used the Fastai library for this notbook

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


# In[ ]:


train = pd.read_excel('../input/Train_Data.xlsx')


# In[ ]:


import fastai
from fastai.text import *
from fastai.callbacks import *


# In[ ]:


# check the contents of the dat set
train.head()


# In[ ]:


train['question_text'][2]


# In[ ]:


train['question_text'][9]


# In[ ]:


train['question_text'][19]


# In[ ]:


from sklearn.model_selection import train_test_split
train, val = train_test_split(train)


# In[ ]:


# Classifier model data
data_clas  = TextClasDataBunch.from_df('.', train,valid_df=val,text_cols='question_text',label_cols='target')


# In[ ]:


data_clas.show_batch()


# In[ ]:


learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(2,1.74E-01)


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(3, 1e-05)


# In[ ]:


txt_ci = TextClassificationInterpretation.from_learner(learn)


# **Some Cool Visualization to interpret the model working**
# 
# Provides an interpretation of classification based on input sensitivity.
# 
# The darker the word-shading in the below example, the more it contributes to the classification. Results here are without any fitting. After fitting to acceptable accuracy, this class can show you what is being used to produce the classification of a particular case.

# In[ ]:


import matplotlib.cm as cm
test_text = "Bangalore was perhaps the best place i have ever seen!"
txt_ci.show_intrinsic_attention(test_text,cmap=cm.Purples)


# we see that the first 4 tokens are the most influential for this sentence

# we dont find a huge difference between AWD LSTM and Language models because the text corpus is generic, its similar to the corpus which the embeddings are trained on like Wiki103 corpus, but Language models give a huge lift in accuracy/F1 when the corpus is domain specific and also when we have a large amount of domain corpus which need not be labelled

# In[ ]:




