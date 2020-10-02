#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import fastai
from fastai import *
from fastai.text import *
from fastai.core import *
fastai.__version__


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook

import os
print(os.listdir("../input"))


# # Data

# In[ ]:


train_df = pd.read_csv('../input/train_data.txt',  sep=':::', header=None, 
                       names=('name', 'genre', 'text'), index_col=0)


# In[ ]:


train_df.head()


# In[ ]:


train_df.info()


# In[ ]:


train_df.genre.value_counts()


# In[ ]:


test_df = pd.read_csv('../input/test_data.txt',  sep=':::', header=None, 
                      names=('name', 'text'), index_col=0)
test_df.head()


# # Preprocessing

# ### Lower

# In[ ]:


train_df['text'] = train_df['text'].str.lower()
test_df['text'] = test_df['text'].str.lower()


# ### train_test_split

# In[ ]:


# split data into training and validation set
df_trn, df_val = train_test_split(train_df[['genre','text']], stratify = train_df['genre'], 
                                  test_size = 0.05, random_state = 42)
df_trn.shape, df_val.shape


# In[ ]:


df_trn.head()


# # Encoder

# In[ ]:


# Language model data
data_lm = TextLMDataBunch.from_df('.', train_df=df_trn, valid_df=df_val)
data_lm.save('tmp_lm.pkl')


# In[ ]:


learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(3,)
learn.freeze()


# In[ ]:


learn.save_encoder('ft_enc')


# # Text Classification

# In[ ]:


# Classifier model data
data_clas = TextClasDataBunch.from_df(path = '.', train_df=df_trn, valid_df=df_val, test_df=test_df,
                                      text_cols='text', label_cols='genre',
                                      vocab=data_lm.vocab, bs=32, shuffle = False,)
data_clas.save('tmp_DB.pkl')


# In[ ]:


data_clas


# In[ ]:


learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
learn.load_encoder('ft_enc')


# ### step-1

# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(10, 1e-2)
learn.save('step-1')


# In[ ]:


learn.recorder.plot()
learn.recorder.plot_losses()
learn.recorder.plot_metrics()


# ### step-2

# In[ ]:


learn.freeze_to(-2)
learn.fit_one_cycle(10, slice(5e-3/2., 1e-3))
learn.freeze()
learn.save('step-2')


# In[ ]:


learn.recorder.plot()
learn.recorder.plot_losses()
learn.recorder.plot_metrics()


# ### step-3

# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(10, slice(1e-3/5, 1e-4))
learn.freeze()
learn.save('step-3')


# In[ ]:


learn.recorder.plot()
learn.recorder.plot_losses()
learn.recorder.plot_metrics()


# In[ ]:


learn.export()


# # Predict & Submission

# In[ ]:


get_ipython().run_cell_magic('time', '', 'predict = []\npredict_proba = []\nfor text in tqdm_notebook(test_df.text):\n    predicts = learn.predict(text)\n    predict.append(int(predicts[1]))\n    predict_proba.append(np.array(predicts[2]))')


# In[ ]:


predict_proba = pd.DataFrame(predict_proba)
predict_proba.columns = data_clas.classes
predict_proba.to_csv('predict_proba.csv', index=False)
predict_proba.head()


# In[ ]:


genre_sub=[]
for i in predict:
    genre_sub.append(data_clas.classes[i].replace(' ', ''))

submission = pd.DataFrame({'id':range(1, len(predict)+1), 'genre':genre_sub}, columns=['id', 'genre'])
submission.to_csv('submission.csv', index=False)
submission.head()


# In[ ]:


# v2

