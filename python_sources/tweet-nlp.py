#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


from fastai.text import *


# In[ ]:


path_train = '/kaggle/input/nlp-getting-started/train.csv'
path_test = '/kaggle/input/nlp-getting-started/test.csv'


# In[ ]:


train_df = pd.read_csv(path_train)
test_df = pd.read_csv(path_test)


# In[ ]:


train_df.head()


# In[ ]:


train_df['text']


# In[ ]:


train_lm = pd.concat([train_df[['text']], test_df[['text']]])


# In[ ]:


train_lm.shape


# In[ ]:


train_lm.head()


# In[ ]:


data_lm = (TextList.from_df(train_lm).split_by_rand_pct(0.10).label_for_lm().databunch(bs=128))
data_lm.save('tmp_lm')


# In[ ]:


data_lm.show_batch()


# In[ ]:


data_lm.vocab.itos[:10]


# In[ ]:


data_lm.train_ds[0]


# In[ ]:


data_lm.train_ds[0][0].data[:10]


# In[ ]:


learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(1,1e-02,moms=(0.8,0.7))


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(4,1e-02)


# In[ ]:


learn.save('learn1')


# In[ ]:


learn.load('learn1')


# In[ ]:


TEXT = "Just got sent this"
N_WORDS = 10
N_SENTENCES = 1


# In[ ]:


print("\n".join(learn.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))


# In[ ]:


learn.save_encoder('fine_tuned_enc')


# In[ ]:


data_cls = (TextList.from_df(train_df[['text', 'target']], vocab=data_lm.vocab).split_by_rand_pct(valid_pct = 0.1).label_from_df('target').databunch(bs=32))


# In[ ]:


learn = text_classifier_learner(data_cls, AWD_LSTM, drop_mult=0.5, metrics=[accuracy, FBeta(beta=1)])


# In[ ]:


learn.load_encoder('fine_tuned_enc')


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))


# In[ ]:


learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))


# In[ ]:


learn.save('first')


# In[ ]:


learn.load('first')


# In[ ]:


learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))


# In[ ]:


learn.save('second')


# In[ ]:


learn.load('second')


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(10, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))


# In[ ]:


pred = []
for i in test_df['text']:
    pred.append(np.argmax(np.array(learn.predict(i)[2])))


# In[ ]:


test_df['target'] = pred
test_df.head()


# In[ ]:


test_pd = test_df[['id','target']]


# In[ ]:


test_pd.head()


# In[ ]:


test_pd.to_csv('submission.csv',index=False)


# In[ ]:




