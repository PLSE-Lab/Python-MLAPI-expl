#!/usr/bin/env python
# coding: utf-8

# In my two previous kernels I created distance features and imputed molecular features on the test data. I now combine these with estimated Mulliken charges computed using Open Babel in a third-party kernel. There is an incremental improvement in predictive performance. 

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


from numpy.random import permutation
from sklearn import metrics
import lightgbm


# In[ ]:


train = pd.read_csv('../input/champs-scalar-coupling/train.csv')
test = pd.read_csv('../input/champs-scalar-coupling/test.csv')
sub = pd.read_csv('../input/champs-scalar-coupling/sample_submission.csv')


# In[ ]:


# previously constructed distance features
# https://www.kaggle.com/robertburbidge/distance-features
train_dist = pd.read_csv('../input/distance-features/train_dist.csv')
test_dist = pd.read_csv('../input/distance-features/test_dist.csv')
train = pd.merge(train.drop(['atom_index_0', 'atom_index_1', 'type'], axis=1), train_dist, how='left', on='id')
test = pd.merge(test.drop(['atom_index_0', 'atom_index_1', 'type'], axis=1), test_dist, how='left', on='id')
del train_dist, test_dist


# In[ ]:


# previously imputed molecular features
# https://www.kaggle.com/robertburbidge/imputing-molecular-features
train_dipole_moment = pd.read_csv('../input/imputing-molecular-features/train_dipole_moment.csv')
test_dipole_moment = pd.read_csv('../input/imputing-molecular-features/test_dipole_moment.csv')
train = pd.merge(train, train_dipole_moment, how='left', on='molecule_name')
test = pd.merge(test, test_dipole_moment, how='left', on='molecule_name')
train_potential_energy = pd.read_csv('../input/imputing-molecular-features/train_potential_energy.csv')
test_potential_energy = pd.read_csv('../input/imputing-molecular-features/test_potential_energy.csv')
train = pd.merge(train, train_potential_energy, how='left', on='molecule_name')
test = pd.merge(test, test_potential_energy, how='left', on='molecule_name')


# In[ ]:


# mulliken charges with Open Babel
# https://www.kaggle.com/asauve/v7-estimation-of-mulliken-charges-with-open-babel
train_ob_charges = pd.read_csv('../input/v7-estimation-of-mulliken-charges-with-open-babel/train_ob_charges.csv')
test_ob_charges = pd.read_csv('../input/v7-estimation-of-mulliken-charges-with-open-babel/test_ob_charges.csv')
train = pd.merge(train, train_ob_charges[['molecule_name', 'atom_index', 'eem']], how='left',
         left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index']).\
    rename({'eem': 'eem0'}, axis=1)
train = pd.merge(train, train_ob_charges[['molecule_name', 'atom_index', 'eem']], how='left',
         left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index']).\
    rename({'eem': 'eem1'}, axis=1)
test = pd.merge(test, test_ob_charges[['molecule_name', 'atom_index', 'eem']], how='left',
         left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index']).\
    rename({'eem': 'eem0'}, axis=1)
test = pd.merge(test, test_ob_charges[['molecule_name', 'atom_index', 'eem']], how='left',
         left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index']).\
    rename({'eem': 'eem1'}, axis=1)


# In[ ]:


# https://www.kaggle.com/artgor/artgor-utils
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                start_mem - end_mem) / start_mem))
    return df


# In[ ]:


train = reduce_mem_usage(train)
test = reduce_mem_usage(test)


# In[ ]:


# features for prediction
pred_vars = [v for v in train.columns if v not in ['id', 'molecule_name', 'scalar_coupling_constant',
                                                   'atom_index_x', 'atom_index_y']]


# In[ ]:


# train-val split by molecule_name
molecule_names = pd.DataFrame(permutation(train['molecule_name'].unique()),columns=['molecule_name'])
nm = molecule_names.shape[0]
ntrn = int(0.9*nm)
nval = int(0.1*nm)

tmp_train = pd.merge(train, molecule_names[0:ntrn], how='right', on='molecule_name')
tmp_val = pd.merge(train, molecule_names[ntrn:nm], how='right', on='molecule_name')

X_train = tmp_train[pred_vars]
X_val = tmp_val[pred_vars]
y_train = tmp_train['scalar_coupling_constant']
y_val = tmp_val['scalar_coupling_constant']
del tmp_train, tmp_val


# In[ ]:


# heuristic parameters for LightGBM
params = { 'objective': 'regression_l1',
           'learning_rate': 0.1,
           'num_leaves': 1023,
           'num_threads': -1,
           'bagging_fraction': 0.5,
           'bagging_freq': 1,
           'feature_fraction': 0.9,
           'lambda_l1': 10.0,
           'max_bin': 255,
           'min_child_samples': 15,
           }


# In[ ]:


# categorical features picked up from train_dist, already integer-encoded
cat_feats = ['type', 'type_0', 'type_1', 'atom_0l', 'atom_0r', 'atom_1l', 'atom_1r']


# In[ ]:


# data for lightgbm
train_data = lightgbm.Dataset(X_train, label=y_train, categorical_feature=cat_feats)
val_data = lightgbm.Dataset(X_val, label=y_val, categorical_feature=cat_feats)


# In[ ]:


# training
model = lightgbm.train(params,
                       train_data,
                       valid_sets=[train_data, val_data], verbose_eval=500,
                       num_boost_round=4000,
                       early_stopping_rounds=100)


# In[ ]:


# evaluation metric for validation
# https://www.kaggle.com/abhishek/competition-metric
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


# validation
preds = model.predict(X_val)
metric(pd.concat([X_val, y_val], axis=1), preds)


# In[ ]:


# submission
preds_sub = model.predict(test[pred_vars])
sub['scalar_coupling_constant'] = preds_sub
sub.to_csv('submission_feats_dist_mol_mc01.csv', index=False)

