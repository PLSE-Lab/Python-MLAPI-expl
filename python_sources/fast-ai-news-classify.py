#!/usr/bin/env python
# coding: utf-8

# https://mc.ai/tutorial-on-text-classification-nlp-using-ulmfit-and-fastai-library-in-python/

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# In[ ]:


# import libraries 
import fastai 
from fastai import * 
from fastai.text import * 
import pandas as pd 
import numpy as np 
from functools import partial 
import io 
import os


# In[ ]:


from sklearn.datasets import fetch_20newsgroups 


# In[ ]:


dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove= 
                             ('headers', 'footers', 'quotes'))


# In[ ]:


documents = dataset.data


# In[ ]:


df = pd.DataFrame({'label':dataset.target, 'text':dataset.data})


# In[ ]:


df.shape


# In[ ]:


df = df[df['label'].isin([1,10])] 
df = df.reset_index(drop = True)


# In[ ]:


df['label'].value_counts()


# In[ ]:


df['text'] = df['text'].str.replace("[^a-zA-Z]", " ")


# In[ ]:


import nltk
nltk.download('stopwords') 
from nltk.corpus import stopwords 


# In[ ]:


stop_words = stopwords.words('english')


# In[ ]:


# tokenization 
tokenized_doc = df['text'].apply(lambda x: x.split()) 


# In[ ]:


# remove stop-words 
tokenized_doc = tokenized_doc.apply(lambda x:[item for item in x if 
                                    item not in stop_words]) 


# In[ ]:


# de-tokenization 
detokenized_doc = [] 


# In[ ]:


for i in range(len(df)):
    t =' '.join(tokenized_doc[i]) 
    detokenized_doc.append(t) 


# In[ ]:


df['text'] = detokenized_doc


# In[ ]:


from sklearn.model_selection import train_test_split 


# In[ ]:


# split data into training and validation set 
df_trn, df_val = train_test_split(df, stratify = df['label'], 
                                  test_size = 0.4, 
                                  random_state = 12)


# In[ ]:


df_trn.shape, df_val.shape


# In[ ]:


# Language model data 
data_lm = TextLMDataBunch.from_df(train_df = df_trn, valid_df = 
                                  df_val, path = "") 


# In[ ]:


# Classifier model data 
data_clas = TextClasDataBunch.from_df(path = "", train_df = df_trn, 
                                      valid_df = df_val,  
                                      vocab=data_lm.train_ds.vocab, 
                                      bs=32)


# In[ ]:


learn = language_model_learner(data_lm, pretrained_model=URLs.WT103,  
                               drop_mult=0.7)


# In[ ]:


# train the learner object with learning rate = 1e-2 
learn.fit_one_cycle(3, 1e-2)


# In[ ]:


learn.save_encoder('ft_enc')


# In[ ]:


learn = text_classifier_learner(data_clas, drop_mult=0.7) 
learn.load_encoder('ft_enc')


# In[ ]:


learn.fit_one_cycle(1, 1e-2)


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(2, slice(2e-3/100, 2e-3))


# In[ ]:


# get predictions 
preds, targets = learn.get_preds(DatasetType.Valid) 
predictions = np.argmax(preds, axis = 1) 


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=2)
#predictions = model.predict(X_test, batch_size=1000)

LABELS = ['graphics','hockey'] 

confusion_matrix = metrics.confusion_matrix(targets, predictions)

plt.figure(figsize=(5, 5))
sns.heatmap(confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d", annot_kws={"size": 20});
plt.title("Confusion matrix", fontsize=20)
plt.ylabel('True label', fontsize=20)
plt.xlabel('Predicted label', fontsize=20)
plt.show()


# In[ ]:




