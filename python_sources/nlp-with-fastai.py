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


from fastai import *
from fastai.text import *


# In[ ]:


path = "/kaggle/input/nlp-getting-started/"
train = pd.read_csv(f'{path}train.csv')
test = pd.read_csv(f'{path}test.csv')


# In[ ]:


train.head()


# In[ ]:


train.target.plot.hist()


# In[ ]:


train.drop(['id','keyword','location'],axis = 1,inplace=True)
test.copy().drop(['id','keyword','location'],axis = 1,inplace=True)


# In[ ]:


test['target'] = 0


# In[ ]:


valid_x = np.random.choice(np.arange(len(train)),round(len(train)*0.2),replace=False)
train_x = np.asarray(list(set(np.arange(len(train)))-set(valid_x)))

assert(train.shape[0] == train_x.shape[0]+valid_x.shape[0])


# In[ ]:


bs=32 #batch size, keep lower in case of low ram


# In[ ]:


df_tr = train.iloc[train_x,[1,0]]
df_val = train.iloc[valid_x,[1,0]]

df_te = test.iloc[:,[1,0]]


# In[ ]:


data_lm = TextLMDataBunch.from_df(path,train_df = df_tr,valid_df = df_val)

data_class = TextClasDataBunch.from_df(path,train_df = df_tr,valid_df = df_val,vocab=data_lm.train_ds.vocab, bs=bs,test_df = df_te)


# In[ ]:


data_lm.show_batch()


# In[ ]:


x,y = next(iter(data_lm.train_dl))
example = x[:15,:15].cpu()
texts = pd.DataFrame([data_lm.train_ds.vocab.textify(l).split(' ') for l in example])
texts


# In[ ]:


torch.cuda.is_available()
torch.cuda.current_device()
torch.cuda.set_device(0)


# In[ ]:


opath = '/kaggle/working'


# In[ ]:


learn = language_model_learner(data_lm,AWD_LSTM,drop_mult=0.1,path = opath)


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(1,1e-2)


# In[ ]:


learn.save('ft')
learn.save_encoder('ft_enc')


# In[ ]:


learner = text_classifier_learner(data_class,AWD_LSTM,drop_mult=0.3)
learner = learner.load_encoder(os.path.join(opath,'models','ft_enc'))


# In[ ]:


learner.lr_find()


# In[ ]:


learner.recorder.plot()


# In[ ]:


learner.freeze_to(-2)
learner.fit_one_cycle(3,slice(1e-02,1e-01/2))


# In[ ]:


learner.unfreeze()
learner.fit_one_cycle(3,slice(1e-1))


# In[ ]:


valid_probs,valid_y = learn.get_preds(ds_type= DatasetType.Valid,ordered = True)


# In[ ]:


valid_probs.numpy()


# In[ ]:


valid_preds = np.argmax(valid_probs.numpy(),1)


# In[ ]:


np.unique(valid_preds,return_counts=True)


# In[ ]:


from sklearn.metrics import *


# In[ ]:


accuracy_score(valid_y,valid_preds)


# In[ ]:


np.unique(valid_preds,return_counts=True)


# In[ ]:


print(learn.predict("huge fire in the neighbour hood:)"))


# In[ ]:


test_probs,_  = learn.get_preds(ds_type = DatasetType.Test,ordered =True)


# In[ ]:


torch.unique(test_probs,return_counts=True)


# In[ ]:


test_preds = np.argmax(test_probs.numpy(),1)
print(f'{test_preds.sum()} positives in {test_preds.shape[0]}')


# In[ ]:


test_probs


# In[ ]:


np.unique(test_preds,return_counts=True)


# In[ ]:


sample_submission = pd.read_csv(os.path.join(path,'sample_submission.csv'))


# In[ ]:


sample_submission['target'] = test_preds
sample_submission.head()


# In[ ]:


sample_submission.to_csv('sub_4.csv',index=False)

