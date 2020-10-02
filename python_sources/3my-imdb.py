#!/usr/bin/env python
# coding: utf-8

# # Preparing the data

# In[ ]:


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


# In[ ]:


# Set your own project id here
# PROJECT_ID = 'your-google-cloud-project'
  
# from google.cloud import bigquery
# client = bigquery.Client(project=PROJECT_ID, location="US")

get_ipython().run_line_magic('matplotlib', 'inline')
from fastai.text import * 


# In[ ]:


print(data.vocab.itos[:12])


# In[ ]:


# but the underlying data is all numbers
data.train_ds[2][0].data[:10]


# In[ ]:


# with the data block API will be more flexible


# # Language model

# In[ ]:


path = untar_data(URLs.IMDB)
path.ls()


# In[ ]:


get_ipython().system(" cat '/tmp/.fastai/data/imdb/README'")


# In[ ]:


(path/'train').ls()


# In[ ]:


(path/'train'/'pos').ls()[:5]


# In[ ]:


get_ipython().system(" cat '/tmp/.fastai/data/imdb/train/pos/10731_7.txt'")


# In[ ]:


path.ls()


# In[ ]:


# language model can use a lot of GPU, may need to decrease batchsize here
# bs = 64
bs = 32


# In[ ]:


data_lm = (TextList.from_folder(path)
          .filter_by_folder(include=['train','test','unsup'])
          .split_by_rand_pct(0.1)
          .label_for_lm()
          .databunch(bs=bs))


# In[ ]:


data_lm.save('data_lm.pkl')


# In[ ]:


data_lm = load_data(path, 'data_lm.pkl', bs=bs)


# In[ ]:


data_lm.show_batch()


# In[ ]:


print(data_lm.vocab.itos[:12])


# In[ ]:


data_lm.train_ds[1][0]


# In[ ]:


data_lm.train_ds[1][0].data[:10]


# In[ ]:


learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot(skip_end=15)


# In[ ]:


learn.fit_one_cycle(1, 1e-2, moms=(0.8, 0.7))


# In[ ]:


learn.save('/kaggle/working/fit_head')


# In[53]:


learn.save('fit_head')


# In[49]:


TEXT = 'I liked this moive because'
N_WORDS = 40


# In[ ]:


# result before fine-tune
for _ in range(2):
    print(learn.predict(TEXT, N_WORDS, temperature=0.75))
    print('\n')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(4, 1e-3, moms=(0.8,0.7))


# In[50]:


learn.save_encoder('/kaggle/working/fine_tuned_enc')


# In[67]:


learn.save_encoder('fine_tuned_enc')


# In[ ]:





# # Classifier

# In[51]:


path = untar_data(URLs.IMDB)


# In[64]:


path.ls()


# In[58]:


data_lm.vocab.itos[:4]


# In[59]:


data_class = (TextList.from_folder(path, vocab=data_lm.vocab)
             .split_by_folder(valid='test')
             .label_from_folder(classes=['neg','pos'])
             .databunch(bs=bs))


# In[60]:


data_class.save('data_class.pkl')


# In[61]:


data_class = load_data(path, 'data_class.pkl', bs=bs)


# In[62]:


data_class.show_batch()


# In[68]:


learn = text_classifier_learner(data_class, AWD_LSTM, drop_mult=0.5)
learn.load_encoder('fine_tuned_enc')


# In[69]:


learn.lr_find()


# In[70]:


learn.recorder.plot()


# In[71]:


learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))


# In[72]:


learn.predict("I really loved that movie, it was awesome!")

