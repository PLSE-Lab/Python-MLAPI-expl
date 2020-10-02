#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import random
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from matplotlib import pyplot as plt

from fastai import *
from fastai.tabular import *
from fastai.basic_data import DataBunch
from tqdm import tqdm_notebook

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ## Data Load and Exploration

# In[ ]:


indir = '../input'


# In[ ]:


df = pd.read_csv(os.path.join(indir, 'train.csv'))
df.describe()


# In[ ]:


test_df = pd.read_csv(os.path.join(indir, 'test.csv')).set_index('ID_code')
test_df.describe()


# In[ ]:


# get the features list
features = [feature for feature in df.columns if 'var' in feature]
len(features)


# Some more features

# In[ ]:


augmented_features = ['min', 'mean', 'max', 'median', 'std', 'abs_mean', 'abs_median', 'abs_std', 'skew', 'kurt', 'sq_kurt']


# In[ ]:


def augment_df(df):
    for feature in features:
        df[f'sq_{feature}'] = df[feature]**2
        df[f'repo_{feature}'] = df[feature].apply(lambda x: 0 if x==0 else 1/x)
        df[f'repo_sq_{feature}'] = df[f'sq_{feature}'].apply(lambda x: 0 if x==0 else 1/x)
    
    df['min'] = df[features].min(axis=1)
    df['mean'] = df[features].mean(axis=1)
    df['max'] = df[features].max(axis=1)
    df['median'] = df[features].median(axis=1)
    df['std'] = df[features].std(axis=1)
    df['var'] = df[features].var(axis=1)
    df['abs_mean'] = df[features].abs().mean(axis=1)
    df['abs_median'] = df[features].abs().median(axis=1)
    df['abs_std'] = df[features].abs().std(axis=1)
    df['skew'] = df[features].skew(axis=1)
    df['kurt'] = df[features].kurt(axis=1)
    
    df['sq_kurt'] = df[[f'sq_{feature}' for feature in features]].kurt(axis=1)
    


# In[ ]:


get_ipython().run_cell_magic('time', '', 'augment_df(df)\ndf.head()')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'augment_df(test_df)\ntest_df.head()')


# In[ ]:


features = features + [f'sq_{feature}' for feature in features] + [f'repo_{feature}' for feature in features] + [f'repo_sq_{feature}' for feature in features]
len(features) 


# Split training data into train and validation sets

# In[ ]:


random.seed(2019)
valid_idx = random.sample(list(df.index.values), int(len(df)*0.05) )


# In[ ]:


# verify that positive sample distribution in validation set is similar to that of the whole data
df.iloc[valid_idx].target.sum() / len(valid_idx) , df.target.sum() / len(df)


# In[ ]:


class roc(Callback):
    '''
    ROC_AUC metric callback for fastai. Compute ROC score over each batch and returns the average over batches.
    TO DO: rolling average
    '''
    def on_epoch_begin(self, **kwargs):
        self.total = 0
        self.batch_count = 0
    
    def on_batch_end(self, last_output, last_target, **kwargs):
        preds = F.softmax(last_output, dim=1)
        # roc_auc_score does not work on batches which does not contain both classes.
        try:
            roc_score = roc_auc_score(to_np(last_target), to_np(preds[:,1]))
            self.total += roc_score
            self.batch_count += 1
        except:
            pass
    
    def on_epoch_end(self, num_batch, **kwargs):
        self.metric = self.total/self.batch_count


# ## FastAI Tabular Learner
# We start off with the default learner from FastAI

# In[ ]:


BATCH_SIZE = 2048


# In[ ]:


def get_data_learner(train_df, train_features, valid_idx, 
                     lr=0.02, epochs=1, layers=[256], ps=[0.2], name='learner'):
    data = TabularDataBunch.from_df(path='.', df=train_df, 
                                    dep_var='target', 
                                    valid_idx=valid_idx, 
                                    cat_names=[], 
                                    cont_names=train_features, 
                                    bs=BATCH_SIZE,
                                    procs=[Normalize],
                                    test_df=test_df)
    learner = tabular_learner(data, layers=layers, ps=ps, metrics=[accuracy, roc()], use_bn=True)
    return learner, data


# In[ ]:


learner, data = get_data_learner(df, features + augmented_features, valid_idx)


# In[ ]:


learner.fit_one_cycle(1, 1e-2)


# In[ ]:


learner.lr_find()
learner.recorder.plot()


# In[ ]:


learner.fit_one_cycle(3, 1e-3)


# In[ ]:


learner.lr_find()
learner.recorder.plot()


# In[ ]:


learner.fit_one_cycle(5, 1e-5)


# In[ ]:


learner.lr_find()
learner.recorder.plot()


# In[ ]:


learner.fit_one_cycle(1, 1e-5)


# In[ ]:


learner.lr_find()
learner.recorder.plot()


# In[ ]:


learner.fit_one_cycle(1, 1e-6)


# ### visualize validation roc

# In[ ]:


# roc_auc_score on validation set
valid_predictions = np.squeeze(to_np(learner.get_preds()[0]))[:, 1]
average_valid_predicts = valid_predictions
valid_auc_score = roc_auc_score(df.iloc[valid_idx].target, average_valid_predicts); valid_auc_score


# In[ ]:


from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_true=df.iloc[valid_idx].target,y_score=average_valid_predicts)
plt.figure(figsize=(9,9))
plt.plot(fpr, tpr)
plt.show()


# ## Test and Submit

# In[ ]:


predictions = np.squeeze(to_np(learner.get_preds(DatasetType.Test)[0]))[:,1]
test_df['target'] = predictions


# In[ ]:


test_df[['target']].to_csv(f'submission_fastai.csv')


# In[ ]:


test_df[['target']].head()


# In[ ]:




