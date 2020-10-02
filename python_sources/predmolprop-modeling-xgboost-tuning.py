#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import time
import gc

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


dtypes = {'atom_index_0':'uint8', 
          'atom_index_1':'uint8', 
          'scalar_coupling_constant':'float32', 
          'num_C':'uint8', 
          'num_H':'uint8', 
          'num_N':'uint8', 
          'num_O':'uint8', 
          'num_F':'uint8',
          'total_atoms':'uint8',
          'num_bonds':'uint8', 
          'num_mol_bonds':'uint8', 
          'min_d':'float32', 
          'mean_d':'float32', 
          'max_d':'float32', 
          'space_dr':'float32', 
          'bond_dr':'float32',
          'bond_1':'uint8', 
          'bond_2':'uint8', 
          'bond_3':'uint8', 
          'atom_0_pc':'float32', 
          'atom_end_pc':'float32',
          'atom_2_hyb':'uint8', 
          'atom_3_hyb':'uint8', 
          'atom_end_hyb':'uint8', 
          'path_count':'uint8', 
          'atom_0_min':'float32',
          'atom_0_mean':'float32', 
          'atom_0_max':'float32', 
          'atom_0_Cmin':'float32', 
          'atom_0_Cmean':'float32',
          'atom_0_Cmax':'float32', 
          'atom_0_Omin':'float32', 
          'atom_0_Omean':'float32',
          'atom_0_Omax':'float32', 
          'atom_0_Nmin':'float32', 
          'atom_0_Nmean':'float32', 
          'atom_0_Nmax':'float32',
          'atom_0_Fmin':'float32', 
          'atom_0_Fmean':'float32', 
          'atom_0_Fmax':'float32', 
          'atom_end_min':'float32',
          'atom_end_mean':'float32', 
          'atom_end_max':'float32', 
          'atom_end_Cmin':'float32', 
          'atom_end_Cmean':'float32',
          'atom_end_Cmax':'float32', 
          'atom_end_Omin':'float32', 
          'atom_end_Omean':'float32',
          'atom_end_Omax':'float32', 
          'atom_end_Nmin':'float32', 
          'atom_end_Nmean':'float32', 
          'atom_end_Nmax':'float32',
          'atom_end_Fmin':'float32', 
          'atom_end_Fmean':'float32', 
          'atom_end_Fmax':'float32',
          'Dmin_COM':'float32', 
          'Dmean_COM':'float32', 
          'Dmax_COM':'float32',
          'COM_dr_0': 'float32',
          'COM_dr_1': 'float32',
          'bond2_angle': 'float32',
          'bond3_angle': 'float32'
         }


# In[ ]:


train = pd.read_csv("../input/predmolprop-featureengineering-final/train_extend.csv",dtype=dtypes)
test = pd.read_csv("../input/predmolprop-featureengineering-finaltest/test_extend.csv",dtype=dtypes)


# In[ ]:


PopList = ['molecule_name', 'atom_index_0','atom_index_1','num_bonds','atom_end_type',
           'atom_2_hyb','atom_3_hyb','atom_end_hyb','bond_1']

for col in PopList:
    train.pop(col)
    test.pop(col)
    
train.fillna(value ='',inplace= True)
test.fillna(value='',inplace=True)

coupling_types = sorted(list(train.type.unique()))
gc.collect()


# In[ ]:


train.columns


# In[ ]:


len(train.columns)


# In[ ]:


sns.distplot(train[train.bond2_angle!=-1].bond2_angle)


# In[ ]:


train.loc[train.bond3_angle.map(lambda x: float(x)==type(''))].bond3_angle


# In[ ]:


train.bond3_angle.map(lambda x: float(x))


# In[ ]:


# Encode categorical features
from sklearn.preprocessing import LabelEncoder

cols = ['atom_0_type2','atom_2_type','atom_3_type','atom_end_type2']
for col in cols:
    enc = LabelEncoder()
    train[col]=enc.fit_transform(train[col]).astype(np.uint8)
    test[col]=enc.transform(test[col]).astype(np.uint8)
del cols


# In[ ]:


# feature lists
fc_size = ['num_mol_bonds', 'min_d','mean_d', 'max_d', 'total_atoms',
           'Dmin_COM', 'Dmean_COM', 'Dmax_COM',
           'num_C', 'num_H', 'num_N', 'num_O', 'num_F']

fc_atom_0 = ['atom_0_pc', 'atom_0_type2','COM_dr_0',
             'atom_0_min','atom_0_mean', 'atom_0_max', 
             'atom_0_Cmin', 'atom_0_Cmean','atom_0_Cmax', 
             'atom_0_Omin', 'atom_0_Omean','atom_0_Omax', 
             'atom_0_Nmin', 'atom_0_Nmean', 'atom_0_Nmax',
             'atom_0_Fmin', 'atom_0_Fmean', 'atom_0_Fmax']

fc_atom_end = ['atom_end_pc', 'atom_end_type2', 'COM_dr_1',
               'atom_end_min','atom_end_mean', 'atom_end_max', 
               'atom_end_Cmin', 'atom_end_Cmean','atom_end_Cmax', 
               'atom_end_Omin', 'atom_end_Omean','atom_end_Omax', 
               'atom_end_Nmin', 'atom_end_Nmean', 'atom_end_Nmax',
               'atom_end_Fmin', 'atom_end_Fmean', 'atom_end_Fmax', ]

fc_distance = ['path_count','space_dr', 'bond_dr']

fc1 = set(fc_size+fc_atom_0+fc_atom_end)

fc2 = set(['atom_2_type','bond_2','bond2_angle'])
fc3 = set(['atom_2_type','atom_3_type','bond_2', 'bond_3','bond3_angle'])


# In[ ]:


# Process Data
from sklearn.model_selection import train_test_split
def ProcessData(df,features,test_size=0.25):
    if test_size == 0:
        train_Y = df.pop('scalar_coupling_constant')
        train_type = df.pop('type')
        df.pop('id')
        return df.loc[:,df.columns.map(lambda x: x in features)], train_Y, train_type
    
    train_X, val_X, train_Y, val_Y = train_test_split(df.loc[:,df.columns.map(lambda x: x in features or x=='type')], 
                                                      df.scalar_coupling_constant,test_size=test_size,random_state=42)
    
    train_type = train_X.pop('type')
    val_type = val_X.pop('type')
    return train_X, train_Y, train_type, val_X, val_Y, val_type


# [https://www.kaggle.com/uberkinder/efficient-metric](http://)

# In[ ]:


def CalcLMAE(y_true, y_pred, groups, floor=1e-9):
    maes = (y_true-y_pred).abs().groupby(groups).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()


# In[ ]:


from xgboost import XGBRegressor
from xgboost import XGBRFRegressor
from math import log

def SingleRun(df,features,test_size=0.25,model_fn=XGBRegressor,includeType=False,early_stopping_rounds=None,**kwargs):
    data = ProcessData(df,features,test_size)
    if(test_size==0):
        train_X,train_Y,train_type = data
    else:
        train_X,train_Y,train_type,val_X,val_Y,val_type = data
    if includeType:
        train_X=train_X.join(train_type)
        val_X=val_X.join(val_type)
        enc = LabelEncoder()
        train_X.type=enc.fit_transform(train_X.type).astype(np.uint8)
        val_X.type=enc.transform(val_X.type).astype(np.uint8)
    
    model = model_fn(**kwargs)
    t1=time.time()
    if early_stopping_rounds is None:
        model.fit(train_X,train_Y, verbose=False)
    else:
        if test_size==0:
            print('ERROR: need test data for early_stopping_rounds')
            return -1,-1,-1
        model.fit(train_X,train_Y, early_stopping_rounds=5, eval_set=[(val_X, val_Y)], verbose=False)
        print('best LMAE:',log(model.best_score))
        print('best ntree:', model.best_ntree_limit)
    t2=time.time()
    print('training time:',t2-t1)
    train_predict = pd.Series(model.predict(train_X),index=train_X.index)
    train_LMAE = CalcLMAE(train_Y,train_predict,train_type)
    print('\ttrain LMAE:',train_LMAE)
    if(test_size==0):
        val_LMAE = None
        g = sns.FacetGrid(pd.DataFrame({'type':train_type,'scalar_coupling_constant': train_Y,'predictions':train_predict}), 
                          col="type", col_order = coupling_types,sharex=False,sharey=False)
        g.map(sns.scatterplot, "scalar_coupling_constant","predictions")
    else:
        val_predict = pd.Series(model.predict(val_X),index=val_X.index)
        val_LMAE = CalcLMAE(val_Y,val_predict,val_type)
        print('\tval LMAE:',val_LMAE)
        g = sns.FacetGrid(pd.DataFrame({'type':val_type,'scalar_coupling_constant': val_Y,'predictions':val_predict}), 
                          col="type", col_order = coupling_types,sharex=False,sharey=False)
        g.map(sns.scatterplot, "scalar_coupling_constant","predictions")
    
    gc.collect()
    return model, train_LMAE, val_LMAE


# In[ ]:


# First use a fairly high learning rate
train_sample = train[train.type=='1JHN']
model_1JHN,_,_=SingleRun(train_sample,fc1,test_size=0.2,model_fn=XGBRegressor,includeType=False,early_stopping_rounds=5,
                max_depth=11, learning_rate=0.25, n_estimators=500, 
                verbosity=1, 
                objective='reg:squarederror', booster='gbtree', eval_metric='mae',
                n_jobs=4, 
                gamma=0, min_child_weight=1, max_delta_step=0, 
                subsample=0.8, colsample_bytree=0.2, colsample_bylevel=1, colsample_bynode=1, 
                reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, 
                random_state=0, seed=None, missing=None, importance_type='gain')
# 242 trees appears to be best


# In[ ]:


# Now tune tree specific parameters
# Tune the minimum child weight and the tree depth
train_sample = train[train.type=='1JHN']
child_weight_array = [0, 1, 2, 5, 10]
depth_array = [2, 3, 5, 9]
t_LMAE_inner = []; t_LMAE_outer = []; v_LMAE_inner = []; v_LMAE_outer = [];
for cw in child_weight_array:
    print('For child weight: '+str(cw))
    for depth in depth_array:
        print('For depth: '+str(depth))
        model_1JHN,t_LMAE,v_LMAE=SingleRun(train_sample,fc1,test_size=0.2,model_fn=XGBRegressor,includeType=False,early_stopping_rounds=None,
                max_depth=depth, learning_rate=0.25, n_estimators=242, 
                verbosity=1, 
                objective='reg:squarederror', booster='gbtree', eval_metric='mae',
                n_jobs=4, 
                gamma=0, min_child_weight=cw, max_delta_step=0, 
                subsample=0.8, colsample_bytree=0.2, colsample_bylevel=1, colsample_bynode=1, 
                reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, 
                random_state=0, seed=None, missing=None, importance_type='gain')
        t_LMAE_inner.append(t_LMAE)
        v_LMAE_inner.append(v_LMAE)
    t_LMAE_outer.append(t_LMAE_inner)
    v_LMAE_outer.append(v_LMAE_inner)
        


# In[ ]:


# Child weight = 2 and tree depth = 5 give the lowese LMAE: -0.167
min(min(v_LMAE_outer))


# In[ ]:


# Now tune max features
feature_array = [0.1, 0.2, 0.5, 0.8, 1]
for feature in feature_array:
    print('For num features: '+str(feature))
    model_1JHN,t_LMAE,v_LMAE=SingleRun(train_sample,fc1,test_size=0.2,model_fn=XGBRegressor,includeType=False,early_stopping_rounds=None,
            max_depth=5, learning_rate=0.3, n_estimators=160, 
            verbosity=1, 
            objective='reg:squarederror', booster='gbtree', eval_metric='mae',
            n_jobs=4, 
            gamma=0, min_child_weight=2, max_delta_step=0, 
            subsample=0.8, colsample_bytree=feature, colsample_bylevel=1, colsample_bynode=1, 
            reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, 
            random_state=0, seed=None, missing=None, importance_type='gain')
    t_LMAE_inner.append(t_LMAE)
    v_LMAE_inner.append(v_LMAE)


# In[ ]:


# Best at colsample_bytree = 1 LMAE: -0.355
min(v_LMAE_inner)


# In[ ]:


# Now tune subsample
subsample_array = [0.1, 0.2, 0.5, 0.8, 1]
for subsample in subsample_array:
    print('For subsample: '+str(subsample))
    model_1JHN,t_LMAE,v_LMAE=SingleRun(train_sample,fc1,test_size=0.2,model_fn=XGBRegressor,includeType=False,early_stopping_rounds=None,
            max_depth=5, learning_rate=0.3, n_estimators=160, 
            verbosity=1, 
            objective='reg:squarederror', booster='gbtree', eval_metric='mae',
            n_jobs=4, 
            gamma=0, min_child_weight=2, max_delta_step=0, 
            subsample=subsample, colsample_bytree=1, colsample_bylevel=1, colsample_bynode=1, 
            reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, 
            random_state=0, seed=None, missing=None, importance_type='gain')
    t_LMAE_inner.append(t_LMAE)
    v_LMAE_inner.append(v_LMAE)


# In[ ]:


# Best at subsample = 1 LMAE: -0.3622


# In[ ]:


# Try one-fifth learning rate and 5x trees
model_1JHN,t_LMAE,v_LMAE=SingleRun(train_sample,fc1,test_size=0.2,model_fn=XGBRegressor,includeType=False,early_stopping_rounds=None,
        max_depth=5, learning_rate=0.06, n_estimators=800, 
        verbosity=1, 
        objective='reg:squarederror', booster='gbtree', eval_metric='mae',
        n_jobs=4, 
        gamma=0, min_child_weight=2, max_delta_step=0, 
        subsample=1, colsample_bytree=1, colsample_bylevel=1, colsample_bynode=1, 
        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, 
        random_state=0, seed=None, missing=None, importance_type='gain')
print('LMAE with 1/5th rate (0.06) and 5x trees (800): '+str(v_LMAE))

# Try one-tenth learning rate and 10x trees
model_1JHN,t_LMAE,v_LMAE=SingleRun(train_sample,fc1,test_size=0.2,model_fn=XGBRegressor,includeType=False,early_stopping_rounds=None,
        max_depth=5, learning_rate=0.03, n_estimators=1600, 
        verbosity=1, 
        objective='reg:squarederror', booster='gbtree', eval_metric='mae',
        n_jobs=4, 
        gamma=0, min_child_weight=2, max_delta_step=0, 
        subsample=1, colsample_bytree=1, colsample_bylevel=1, colsample_bynode=1, 
        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, 
        random_state=0, seed=None, missing=None, importance_type='gain')
print('LMAE with 1/5th rate (0.03) and 5x trees (1600): '+str(v_LMAE))


# In[ ]:


# Try old model parameters without k-fold
model_1JHN,t_LMAE,v_LMAE=SingleRun(train_sample,fc1,test_size=0.2,model_fn=XGBRegressor,includeType=False,early_stopping_rounds=None,
        max_depth=11, learning_rate=0.1, n_estimators=3000, 
        verbosity=1, 
        objective='reg:squarederror', booster='gbtree', eval_metric='mae',
        n_jobs=4, 
        gamma=0, min_child_weight=1, max_delta_step=0, 
        subsample=1, colsample_bytree=1, colsample_bylevel=1, colsample_bynode=1, 
        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, 
        random_state=0, seed=None, missing=None, importance_type='gain')
print('LMAE with old parameters without k-fold: '+str(v_LMAE))


# In[ ]:


model_1JHN,t_LMAE,v_LMAE=SingleRun(train_sample,fc1,test_size=0.2,model_fn=XGBRegressor,includeType=False,early_stopping_rounds=None,
        max_depth=11, learning_rate=0.05, n_estimators=6000, 
        verbosity=1, 
        objective='reg:squarederror', booster='gbtree', eval_metric='mae',
        n_jobs=4, 
        gamma=0, min_child_weight=1, max_delta_step=0, 
        subsample=1, colsample_bytree=1, colsample_bylevel=1, colsample_bynode=1, 
        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, 
        random_state=0, seed=None, missing=None, importance_type='gain')
print('LMAE with old parameters without k-fold even slowe: '+str(v_LMAE))


# In[ ]:




