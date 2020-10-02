#!/usr/bin/env python
# coding: utf-8

# There are several datasets provided only for the train data and not the test data. In other kernels I have shown that imputing the dipole moment and potential energy on the test set marginally improved the LB score over a simple model using only structural information:
# https://www.kaggle.com/robertburbidge/imputing-molecular-features
# 
# And that estimated Mulliken charges can be used to further improve the LB score:
# https://www.kaggle.com/robertburbidge/using-estimated-mulliken-charges
# 
# In another notebook I tried to do the same for the magnetic shielding tensor:
# https://www.kaggle.com/robertburbidge/imputing-magnetic-shielding-tensor/
# 
# However, I can't get that kernel to complete on Kaggle compute so I ran it on a VM with 72 CPUs and 144 GB of RAM and found that I couldn't do much better than predicting the median tensor for all atoms. Rather than waste time pursuing this I decided to investigate the relationship between the magnetic shielding tensor of two atoms in a molecule and their scalar coupling constant.
# 
# Since I have no idea what the relationship may be, I'm using symbolic regression. It shows no relationship. As a sanity check, I show that symbolic regression correctly identifies the scalar coupling constant contribution `fc` as being predictive.

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
from fastsr.estimators.symbolic_regression import SymbolicRegression


# In[ ]:


train = pd.read_csv('../input/train.csv')
magnetic_shielding_tensors = pd.read_csv('../input/magnetic_shielding_tensors.csv')
train = pd.merge(train, magnetic_shielding_tensors, how='left',
              left_on=['molecule_name', 'atom_index_0'],
              right_on=['molecule_name', 'atom_index'])

train.rename(columns = { "XX": "XX_0", "YX": "YX_0", "ZX": "ZX_0",
                         "XY": "XY_0", "YY": "YY_0", "ZY": "ZY_0",
                         "XZ": "XZ_0", "YZ": "YZ_0", "ZZ": "ZZ_0" }, inplace=True)

train = pd.merge(train, magnetic_shielding_tensors, how='left',
              left_on=['molecule_name', 'atom_index_1'],
              right_on=['molecule_name', 'atom_index'])

train.rename(columns = { "XX": "XX_1", "YX": "YX_1", "ZX": "ZX_1",
                         "XY": "XY_1", "YY": "YY_1", "ZY": "ZY_1",
                         "XZ": "XZ_1", "YZ": "YZ_1", "ZZ": "ZZ_1" }, inplace=True)
del magnetic_shielding_tensors


# In[ ]:


# one-hot-encode type
train = pd.concat([train, pd.get_dummies(train['type'], prefix='type')], axis=1)


# In[ ]:


# features for predicting scalar coupling constant on train set from magnetic shielding tensors
pred_vars = ['type_1JHC', 'type_1JHN', 'type_2JHC', 'type_2JHH', 'type_2JHN',
             'type_3JHC', 'type_3JHH', 'type_3JHN', 'XX_0', 'YX_0', 'ZX_0', 'XY_0', 'YY_0', 'ZY_0',
             'XZ_0', 'YZ_0', 'ZZ_0', 'XX_1', 'YX_1', 'ZX_1', 'XY_1',
             'YY_1', 'ZY_1', 'XZ_1', 'YZ_1', 'ZZ_1']


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
y_type = tmp_val['type']
del tmp_train, tmp_val


# In[ ]:


# fit symbolic regression on train
sr = SymbolicRegression(ngen=100, pop_size=10)
sr.fit(X_train.values, y_train.values)


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


# performance on val
preds = sr.predict(X_val.values)
metric(pd.concat([X_val, y_val, y_type], axis=1), preds)


# This suggests that there is no relationship between the magnetic shielding tensors and the scalar coupling constant.
# 
# As a sanity check, in the following I check that symbolic regression can find a relationship when it exists by regressing the scalar coupling constant on its contributions.

# In[ ]:


scc = pd.read_csv('../input/scalar_coupling_contributions.csv')
train = pd.merge(train, scc, how='left',
                 on=['molecule_name', 'atom_index_0', 'atom_index_1', 'type'])
del scc


# In[ ]:


# features for predicting scalar coupling constant on train set from its contributions
pred_vars = ['type_1JHC', 'type_1JHN', 'type_2JHC', 'type_2JHH', 'type_2JHN',
             'type_3JHC', 'type_3JHH', 'type_3JHN', 'fc', 'sd', 'pso', 'dso']


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
y_type = tmp_val['type']
del tmp_train, tmp_val


# In[ ]:


# fit symbolic regression on train
sr = SymbolicRegression(ngen=100, pop_size=10)
sr.fit(X_train.values, y_train.values)


# In[ ]:


# performance on val
preds = sr.predict(X_val.values)
metric(pd.concat([X_val, y_val, y_type], axis=1), preds)


# Symbolic regression has picked up a strong relationship. Let's see what it is.

# In[ ]:


sr.print_best_individuals()


# The best individuals are `fc` with minor random variations (introduced by the stochastic search), which is known to be the major contributing factor.
# 
# Conclusion: I am not going to bother with the magnetic shielding tensor.
