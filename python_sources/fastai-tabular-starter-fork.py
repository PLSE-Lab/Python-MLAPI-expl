#!/usr/bin/env python
# coding: utf-8

# I hope you will find this notebook useful for understanding how to implement [fastai's tabular model](https://course.fast.ai/videos/?lesson=4). There is a lot of tuning that can still be done, and the only feature that has been engineered is the distance.
# 
# I made a slight adjustment to the fastai MAE metric, and I have found that the weighted MAE calculated below seems to give a good approximation of the Public LB.
# 
# Thanks and please upvote if you find this kernel useful.
# 
# * forked. + change mean dist feature

# In[1]:


import numpy as np
import pandas as pd
import os
import time
import datetime
import gc
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from IPython.display import HTML
import json
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR, SVR
pd.options.display.precision = 10
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold


# In[2]:


from fastai import *
from fastai.imports import *
from fastai.tabular import *
from fastai.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_absolute_error


# In[3]:


import os
print(os.listdir("../input"))


# In[4]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sub = pd.read_csv('../input/sample_submission.csv')


# In[5]:


train.head()


# In[6]:


structures = pd.read_csv('../input/structures.csv')

def map_atom_info(df, atom_idx):
    df = pd.merge(df, structures, how = 'left',
                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
                  right_on = ['molecule_name',  'atom_index'])
    
    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df

train = map_atom_info(train, 0)
train = map_atom_info(train, 1)

test = map_atom_info(test, 0)
test = map_atom_info(test, 1)


# Einsum distance calculation borrowed from: https://www.kaggle.com/rakibilly/faster-distance-calculation-from-benchmark

# In[7]:


get_ipython().run_cell_magic('time', '', "train_p_0 = train[['x_0', 'y_0', 'z_0']].values\ntrain_p_1 = train[['x_1', 'y_1', 'z_1']].values\ntest_p_0 = test[['x_0', 'y_0', 'z_0']].values\ntest_p_1 = test[['x_1', 'y_1', 'z_1']].values\n\ntr_a_min_b = train_p_0 - train_p_1\nte_a_min_b = test_p_0 - test_p_1\n\ntrain['dist'] = np.sqrt(np.einsum('ij,ij->i', tr_a_min_b, tr_a_min_b))\ntest['dist'] = np.sqrt(np.einsum('ij,ij->i', te_a_min_b, te_a_min_b))")


# difference of atom indexes. 

# In[8]:


train["index_diff"] = np.abs(train["atom_index_0"]-train["atom_index_1"])
test["index_diff"] = np.abs(test["atom_index_0"]-test["atom_index_1"])


# Distance to type mean from : [Molecular Properties EDA and models](https://www.kaggle.com/artgor/molecular-properties-eda-and-models)
# 
# * Modified to also use aggregation of relative distance (difference of atom indexes)

# In[ ]:


train['dist_speedup_to_type_mean'] = train['dist'] / train.groupby(['type',"index_diff"])['dist'].transform('mean')
test['dist_speedup_to_type_mean'] = test['dist'] / test.groupby(['type',"index_diff"])['dist'].transform('mean')


# In[ ]:


for f in ['type', 'atom_0', 'atom_1']:
    lbl = LabelEncoder()
    lbl.fit(list(train[f].values) + list(train[f].values))
    train[f] = lbl.transform(list(train[f].values))
    test[f] = lbl.transform(list(test[f].values))


# Metric calculation from Abhishek's kernel: https://www.kaggle.com/abhishek/competition-metric

# In[ ]:


def metric(df, preds):
    df["prediction"] = preds
    maes = []
    for t in df.type.unique():
        y_true = df[df.type==t].scalar_coupling_constant.values
        y_pred = df[df.type==t].prediction.values
        mae = np.log(metrics.mean_absolute_error(y_true, y_pred))
        maes.append(mae)
    return np.mean(maes)


# In[ ]:


train.head()


# # Start fastai Tabular Learner

# In[ ]:


tr = train.drop(['id', 'molecule_name'], axis=1)
te = test.drop(['id', 'molecule_name'], axis=1)


# In[ ]:


dep_var = 'scalar_coupling_constant'
cat_names = ['atom_index_0', 'atom_index_1', 'type', 'atom_0', 'atom_1']
cont_names = tr.columns.tolist()
cont_names.remove('scalar_coupling_constant')
cont_names = [e for e in cont_names if e not in (cat_names)]
procs = [Categorify, Normalize]


# In[ ]:


np.random.seed(1984)
idx = np.random.randint(0, len(tr), size=np.int(.2*len(tr)))


# In[ ]:


bs = 4096 
data = (TabularList.from_df(tr, 
                            cat_names=cat_names, 
                            cont_names=cont_names, 
                            procs=procs)
                           .split_by_idx(idx)
                           .label_from_df(cols=dep_var)
                           .add_test(TabularList.from_df(te, 
                                                         cat_names=cat_names, 
                                                         cont_names=cont_names))
                           .databunch(bs=bs))


# In[ ]:


data.show_batch(rows=5)


# In[ ]:


data.show_batch(rows=5, ds_type=DatasetType.Valid)


# In[ ]:


data.show_batch(rows=5, ds_type=DatasetType.Test)


# In[ ]:


def mean_absolute_error_fastai(pred:Tensor, targ:Tensor)->Rank0Tensor:
    "Mean absolute error between `pred` and `targ`."
    pred,targ = flatten_check(pred,targ)
    return F.l1_loss(pred, targ)


# In[ ]:


learn = tabular_learner(data, 
                        layers=[1000,500,100], 
                        emb_drop=0.05,
                        ps=(0.001, 0.01, 0.1),
                        metrics=[mean_absolute_error_fastai, rmse], 
                        wd=1e-2).to_fp16()


# In[ ]:


lr_find(learn, start_lr=1e-4, end_lr=10, num_it=100) #, start_lr=1e-2, end_lr=10, num_it=200
learn.recorder.plot()


# In[ ]:


lr = 2e-3
learn.fit_one_cycle(1, lr, wd=0.9)


# In[ ]:


learn.fit_one_cycle(1, lr/4, wd=0.8)


# In[ ]:


learn.fit_one_cycle(3, lr/10, wd=0.8)


# In[ ]:


learn.fit_one_cycle(1, lr/10, wd=0.8)


# In[ ]:


learn.fit_one_cycle(3, lr/20, wd=0.9)


# In[ ]:


learn.fit_one_cycle(3, lr/40, wd=0.8)


# ### Check Metrics

# In[ ]:


val_preds = learn.get_preds(DatasetType.Valid)
y_true = tr.iloc[idx].scalar_coupling_constant
y_preds = val_preds[0][:,0].numpy()
types = tr.iloc[idx].type


# In[ ]:


maes = []
for t in types.unique():
    y_t = pd.Series(y_true[types==t])
    y_p = pd.Series(y_preds[types==t])
    mae = np.log(mean_absolute_error(y_t, y_p))
    maes.append(mae)

np.mean(maes), np.log(mean_absolute_error(y_true, y_preds)), mean_absolute_error(y_true, y_preds)


# # Predict and Submit

# In[ ]:


test_preds = learn.get_preds(DatasetType.Test)
preds = test_preds[0].numpy()


# In[ ]:


sample_submission = pd.read_csv('../input/sample_submission.csv')

benchmark = sample_submission.copy()
benchmark['scalar_coupling_constant'] = preds
benchmark.to_csv('submission.csv', index=False)


# In[ ]:


benchmark.head()

