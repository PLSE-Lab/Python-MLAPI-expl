#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sln
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


train=pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')


# In[3]:


test=pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')


# In[ ]:


train.info()


# In[ ]:


train.head ()


# In[ ]:


train.isnull().sum()


# Lets build a decision tree

# In[ ]:


diff_bias=[]
for i in  train.columns:
    if train[i].isnull().sum()==1399744:
        diff_bias.append(i)
        


# In[ ]:


tree_df=pd.concat([train[diff_bias],train['target']],axis=1)


# In[ ]:


tree_df.head()


# In[ ]:


tree_df.shape


# In[ ]:


tree_df=tree_df.dropna()


# In[ ]:


tree_df.shape


# In[ ]:


tree_df['target'].value_counts()


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn import tree


# In[ ]:


#for visulaization keep depth low
dt=DecisionTreeRegressor(max_depth=4,min_samples_split=10,max_features=10)


# In[ ]:


clf=dt.fit(tree_df.drop('target',axis=1),tree_df['target'])


# In[ ]:


import graphviz 
dot_data = tree.export_graphviz(clf, out_file='tree.dot') 


# In[ ]:


from sklearn.tree import export_graphviz
from IPython.display import SVG


# In[ ]:


from graphviz import Source
from IPython.display import display


# In[ ]:


graph=Source(tree.export_graphviz(clf,feature_names=tree_df.drop('target',axis=1).columns))


# In[ ]:


display(SVG(graph.pipe(format='svg')))


# ## Exploring Text Data

# In[ ]:


train['comment_text'][0]


# In[ ]:


train.head(5)


# In[ ]:


full_text=' '
for i in train[train['article_id']==2006]['comment_text']:
    full_text+=i
    


# In[ ]:


train['comment_text'][4]


# In[ ]:


train['comment_text'][5]


# In[ ]:


train['comment_text'][6]


# In[ ]:


#lets see the effect of gender on toxic comments
train_new=train.dropna()


# In[ ]:


sln.jointplot(train['target'],train['severe_toxicity'])


# In[ ]:


sln.distplot(train_new['female'])


# Using ULMFit

# In[ ]:


#Before checking the cuttoff lets see
#lets clean the data first


# In[4]:


from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer


# In[5]:


stop = stopwords.words('english')
train['comm_clean'] = train['comment_text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


# In[6]:


test['comm_clean'] = test['comment_text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


# In[ ]:


sln.distplot(train['target'])


# In[7]:


# cuttoff-0.5
train['target_cat']=np.where(train['target']>0.5,1,0)


# In[10]:


# import libraries
import fastai
from fastai import *
from fastai.text import * 
from functools import partial


# In[ ]:


df.to_feather('cleaned_df')


# In[8]:


df=train[['comm_clean','target_cat']]


# In[9]:


from sklearn.model_selection import train_test_split

# split data into training and validation set
df_trn, df_val = train_test_split(df, stratify = df['target_cat'], test_size = 0.4, random_state = 12)


# In[ ]:


# Language model data
#data_lm = TextLMDataBunch.from_df(train_df = df_trn, valid_df = df_val, path = "")


# In[11]:


# Classifier model data

data_lm = TextClasDataBunch.from_df('',df_trn,valid_df=df_val, test_df=test, text_cols=['comm_clean'], 
                                    label_cols=['target_cat'])


# In[ ]:


data_lm.save()


# In[12]:


awd_lstm_clas_config = dict(emb_sz=372, n_hid=1000, n_layers=3, pad_token=1, qrnn=False, bidir=False, output_p=0.4,
                       hidden_p=0.2, input_p=0.6, embed_p=0.1, weight_p=0.5)


# In[13]:


learner = text_classifier_learner(data_lm, AWD_LSTM, max_len=210,config=awd_lstm_clas_config, pretrained = False)


# In[14]:


fnames = ['../input/path-new/lstm_wt103.pth','../input/pickle/itos_wt103.pkl']
learner.load_pretrained(*fnames, strict=False)
learner.freeze()


# In[15]:


#learner.fit_one_cycle(1, 1e-3)


# In[16]:


oof = learner.get_preds(ds_type=DatasetType.Valid)


# In[17]:


preds = learner.get_preds(ds_type=DatasetType.Test, ordered=True)


# In[18]:


o = oof[0].cpu().data.numpy()
l = oof[1].cpu().data.numpy()


# In[19]:


p = preds[0].cpu().data.numpy()


# In[20]:


submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv', index_col='id')


# In[21]:


submission['prediction'] = p[:,1]


# In[22]:


submission.to_csv('submission.csv')


# In[ ]:




