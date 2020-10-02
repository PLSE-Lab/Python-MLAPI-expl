#!/usr/bin/env python
# coding: utf-8

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


# **My Code**

# In[ ]:


from fastai import *
from fastai.vision import *
from fastai.tabular import *


# In[ ]:


path = Path('../input/petfinder-adoption-prediction')


# In[ ]:


path_train_img = path/'train_images';
path_test_img = path/'test_images';
path_train = path/'train';
path_test = path/'test'; 
path_train_sentiment = path/'train_sentiment';
path_test_sentiment = path/'test_sentiment';
path_train.ls(), path_test.ls()


# In[ ]:


df_train = pd.read_csv(path_train/'train.csv')
df_train.shape


# In[ ]:


df_test = pd.read_csv(path_test/'test.csv')
df_test.shape


# **Using Fastai Tabular**

# In[ ]:


import json
from pprint import pprint

tuplist = []
for filename in os.listdir(path_train_sentiment):
    if filename.endswith(".json"):
      with open(path_train_sentiment/filename) as f:
        data = json.load(f)
        tuplist.append( (filename[:-5], data['documentSentiment']['magnitude'], data['documentSentiment']['score']) )

tuplist[0]


# In[ ]:


df_sent = pd.DataFrame(tuplist, columns=['PetID', 'magnitude', 'score'])
df_sent.head()


# In[ ]:


df_train_sent = pd.merge(df_train, df_sent, how='left', on='PetID')
df_train_sent.shape


# In[ ]:


df_train_sent.head()


# In[ ]:


# Add in Description Length Column
alist = []
for i in range(len(df_train_sent)):
    alist.append(len(str(df_train_sent.iloc[i]['Description'])))
# Create a column from the list
df_train_sent['desc_len'] = alist
df_train_sent.head()


# In[ ]:


import pprint as pp
with open(path/'train_metadata'/'000fb9572-6.json') as f:
    data = json.load(f)
    x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
    y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
    string = '000fb9572-6.json'
    a = string.find('-')
    print(string[:string.find('-')])
#     pp.pprint(data)
#     pp.pprint(x)
#     pp.pprint(y)


# In[ ]:


# get train image metadata
path_train_img_meta = path/'train_metadata'

tuplist = []
for filename in os.listdir(path_train_img_meta):
    if filename.endswith(".json"):
      with open(path_train_img_meta/filename) as f:
        data = json.load(f)
        x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
        y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']        
        tuplist.append( (filename[:filename.find('-')], x, y) )

tuplist[0]


# In[ ]:


df_meta = pd.DataFrame(tuplist, columns=['PetID', 'vert_x', 'vert_y'])
df_meta.head()


# In[ ]:


# aggregate training data
grouped1 = df_meta.groupby('PetID',as_index=False)['vert_x', 'vert_y'].agg({'vert_min':'min', 'vert_max':'max', 
                                                                               'vert_mean':'mean'})

grouped1.head()


# In[ ]:


grouped1.columns = ['PetID','PetID1','vert_x_min','vert_y_min','PetID2','vert_x_max','vert_y_max','PetID3','vert_x_mean','vert_y_mean']
grouped1['vert_x_mean'] = grouped1['vert_x_mean'].apply(lambda x: int(round(x)))
grouped1['vert_y_mean'] = grouped1['vert_y_mean'].apply(lambda x: int(round(x)))
grouped1 = grouped1[['PetID','vert_x_mean','vert_y_mean']]
grouped1.head()


# In[ ]:


grouped1.shape


# In[ ]:


df_train_meta = pd.merge(df_train_sent, grouped1, how='left', on='PetID')
df_train_meta.shape


# In[ ]:


df_train_meta.head()


# In[ ]:


# for test data
import json
from pprint import pprint

tuplist = []
for filename in os.listdir(path_test_sentiment):
    if filename.endswith(".json"):
      with open(path_test_sentiment/filename) as f:
        data = json.load(f)
        tuplist.append( (filename[:-5], data['documentSentiment']['magnitude'], data['documentSentiment']['score']) )

tuplist[0]


# In[ ]:


df_sent = pd.DataFrame(tuplist, columns=['PetID', 'magnitude', 'score'])
df_sent.head()


# In[ ]:


df_test_sent = pd.merge(df_test, df_sent, how='left', on='PetID')
df_test_sent.shape


# In[ ]:


alist = []
for i in range(len(df_test_sent)):
    alist.append(len(str(df_test_sent.iloc[i]['Description'])))
# Create a column from the list
df_test_sent['desc_len'] = alist
df_test_sent.head()


# In[ ]:


# get test image metadata
path_test_img_meta = path/'test_metadata'

tuplist = []
for filename in os.listdir(path_test_img_meta):
    if filename.endswith(".json"):
      with open(path_test_img_meta/filename) as f:
        data = json.load(f)
        x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
        y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']        
        tuplist.append( (filename[:filename.find('-')], x, y) )

tuplist[0]


# In[ ]:


df_test_meta = pd.DataFrame(tuplist, columns=['PetID', 'vert_x', 'vert_y'])
df_test_meta.head()


# In[ ]:


# aggregate test image metadata
grouped1 = df_test_meta.groupby('PetID',as_index=False)['vert_x', 'vert_y'].agg({'vert_min':'min', 'vert_max':'max', 
                                                                               'vert_mean':'mean'})
grouped1.head()


# In[ ]:


grouped1.columns = ['PetID','PetID1','vert_x_min','vert_y_min','PetID2','vert_x_max','vert_y_max','PetID3','vert_x_mean','vert_y_mean']
grouped1 = grouped1[['PetID','vert_x_min','vert_y_min','vert_x_max','vert_y_max','vert_x_mean','vert_y_mean']]
grouped1.head()


# In[ ]:


df_test_meta = pd.merge(df_test_sent, grouped1, how='left', on='PetID')
df_test_meta.shape


# In[ ]:


# list(df_train_meta.columns.values)
# for col in df_train_meta:
#     print( col, ": ", len(df_train_meta[col].unique()) )


# In[ ]:


dep_var = 'AdoptionSpeed'
cat_names = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3',
             'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'Quantity', 
             'State', 'VideoAmt', 'PhotoAmt'] 
cont_names = ['Age', 'Fee', 'magnitude', 'score', 'desc_len', 'vert_x_mean', 'vert_y_mean']
procs = [FillMissing, Categorify, Normalize]


# In[ ]:


data = (TabularList.from_df(df_train_meta, path='.', cat_names=cat_names, cont_names=cont_names, procs=procs)
                           .random_split_by_pct(0.2, seed=2)
                           .label_from_df(cols=dep_var)
                          .add_test(TabularList.from_df(df_test_meta, path='.', cat_names=cat_names, cont_names=cont_names, procs=procs))
                           .databunch())


# In[ ]:


data.show_batch(rows=10)


# In[ ]:


kappa = KappaScore()
kappa.weights = "quadratic"


# In[ ]:


learn = tabular_learner(data, layers=[1000, 500], ps=[0.001,0.01], emb_drop=0.1, metrics=[accuracy, kappa])


# In[ ]:


learn.fit_one_cycle(5, 1e-2)


# In[ ]:


learn.recorder.plot_losses(last=-1)


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(3, 1e-3)


# In[ ]:


# get predictions for test data
test_preds=learn.get_preds(DatasetType.Test)


# In[ ]:


df_test_meta["AdoptionSpeed"] = test_preds[0].argmax(dim=1)
result = df_test_meta[["PetID","AdoptionSpeed"]]
result.head()


# In[ ]:


result.to_csv("submission.csv", index=False)


# In[ ]:




