#!/usr/bin/env python
# coding: utf-8

# ### V01 is derived from public kernel: https://www.kaggle.com/paulorzp/baseline-ridge/ -runs ElasticNet

# # Libraries

# In[ ]:


import os
import numpy as np
import pandas as pd
import random
import math

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn.metrics import mean_squared_error
import category_encoders as ce

from sklearn.linear_model import Ridge, ElasticNet
from functools import partial
import scipy as sp

import warnings
warnings.filterwarnings("ignore")


# # Utils

# In[ ]:


def seed_everything(seed=777):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


# # Config

# In[ ]:


OUTPUT_DICT = './'

ID = 'Patient_Week'
TARGET = 'FVC'
SEED = 777
seed_everything(seed=SEED)

N_FOLD = 7


# # Data Loading

# In[ ]:


train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
otest = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')


# In[ ]:


# construct train input
train = pd.concat([train,otest])
output = pd.DataFrame()
gb = train.groupby('Patient')
tk0 = tqdm(gb, total=len(gb))
for _, usr_df in tk0:
    usr_output = pd.DataFrame()
    for week, tmp in usr_df.groupby('Weeks'):
        rename_cols = {'Weeks': 'base_Week', 'FVC': 'base_FVC', 'Age': 'base_Age'}
        tmp = tmp.rename(columns=rename_cols)
        drop_cols = ['Age', 'Sex', 'SmokingStatus', 'Percent']
        _usr_output = usr_df.drop(columns=drop_cols).rename(columns={'Weeks': 'predict_Week'}).merge(tmp, on='Patient')
        _usr_output['Week_passed'] = _usr_output['predict_Week'] - _usr_output['base_Week']
        usr_output = pd.concat([usr_output, _usr_output])
    output = pd.concat([output, usr_output])
    
train = output[output['Week_passed']!=0].reset_index(drop=True)


# In[ ]:


# construct test input
test = otest.rename(columns={'Weeks': 'base_Week', 'FVC': 'base_FVC', 'Age': 'base_Age'})
submission = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
submission['Patient'] = submission['Patient_Week'].apply(lambda x: x.split('_')[0])
submission['predict_Week'] = submission['Patient_Week'].apply(lambda x: x.split('_')[1]).astype(int)
test = submission.drop(columns=['FVC', 'Confidence']).merge(test, on='Patient')
test['Week_passed'] = test['predict_Week'] - test['base_Week']
test.set_index('Patient_Week', inplace=True)


# # Prepare folds

# In[ ]:


folds = train[['Patient', TARGET]].copy()
Fold = GroupKFold(n_splits=N_FOLD)
groups = folds['Patient'].values
for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[TARGET], groups)):
    folds.loc[val_index, 'fold'] = int(n)
folds['fold'] = folds['fold'].astype(int)


# # MODEL

# In[ ]:


#===========================================================
# model
#===========================================================
def run_single_model(clf, train_df, test_df, folds, features, target, fold_num=0):
    
    trn_idx = folds[folds.fold!=fold_num].index
    val_idx = folds[folds.fold==fold_num].index
    
    y_tr = target.iloc[trn_idx].values
    X_tr = train_df.iloc[trn_idx][features].values
    y_val = target.iloc[val_idx].values
    X_val = train_df.iloc[val_idx][features].values
    
    oof = np.zeros(len(train_df))
    predictions = np.zeros(len(test_df))
    clf.fit(X_tr, y_tr)
    
    oof[val_idx] = clf.predict(X_val)
    predictions += clf.predict(test_df[features])
    return oof, predictions


def run_kfold_model(clf, train, test, folds, features, target, n_fold=7):
    
    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))
    feature_importance_df = pd.DataFrame()

    for fold_ in range(n_fold):

        _oof, _predictions = run_single_model(clf,
                                              train, 
                                              test,
                                              folds,  
                                              features,
                                              target, 
                                              fold_num=fold_)
        oof += _oof
        predictions += _predictions/n_fold
    
    return oof, predictions


# ## predict FVC

# In[ ]:


target = train[TARGET]
test[TARGET] = np.nan

# features
cat_features = ['Sex', 'SmokingStatus']
num_features = [c for c in test.columns if (test.dtypes[c] != 'object') & (c not in cat_features)]
features = num_features + cat_features
drop_features = [TARGET, 'predict_Week', 'Percent', 'base_Week']
features = [c for c in features if c not in drop_features]

if cat_features:
    ce_oe = ce.OrdinalEncoder(cols=cat_features, handle_unknown='impute')
    ce_oe.fit(train)
    train = ce_oe.transform(train)
    test = ce_oe.transform(test)


# ## Model Trials

# In[ ]:


'''for alpha1 in (1,0.3,0.1,0.03,0.01):
    for l1s in (0.01,0.03,0.1,0.2,0.5,0.8,0.9,0.97,0.99):
        
        print(" For alpha:",alpha1,"& l1_ratio:",l1s)
        clf = ElasticNet(alpha=alpha1, l1_ratio = l1s)
        oof, predictions = run_kfold_model(clf, train, test, folds, features, target, n_fold=N_FOLD)

        train['FVC_pred'] = oof
        test['FVC_pred'] = predictions

        # baseline score
        train['Confidence'] = 100
        train['sigma_clipped'] = train['Confidence'].apply(lambda x: max(x, 70))
        train['diff'] = abs(train['FVC'] - train['FVC_pred'])
        train['delta'] = train['diff'].apply(lambda x: min(x, 1000))
        train['score'] = -math.sqrt(2)*train['delta']/train['sigma_clipped'] - np.log(math.sqrt(2)*train['sigma_clipped'])
        score = train['score'].mean()
        print(score)

        def loss_func(weight, row):
            confidence = weight
            sigma_clipped = max(confidence, 70)
            diff = abs(row['FVC'] - row['FVC_pred'])
            delta = min(diff, 1000)
            score = -math.sqrt(2)*delta/sigma_clipped - np.log(math.sqrt(2)*sigma_clipped)
            return -score

        results = []
        tk0 = tqdm(train.iterrows(), total=len(train))
        for _, row in tk0:
            loss_partial = partial(loss_func, row=row)
            weight = [100]
            result = sp.optimize.minimize(loss_partial, weight, method='SLSQP')
            x = result['x']
            results.append(x[0])

        # optimized score
        train['Confidence'] = results
        train['sigma_clipped'] = train['Confidence'].apply(lambda x: max(x, 70))
        train['diff'] = abs(train['FVC'] - train['FVC_pred'])
        train['delta'] = train['diff'].apply(lambda x: min(x, 1000))
        train['score'] = -math.sqrt(2)*train['delta']/train['sigma_clipped'] - np.log(math.sqrt(2)*train['sigma_clipped'])
        score = train['score'].mean()
        print(score)'''


# In[ ]:


for alpha1 in [0.3]:
    for l1s in [0.8]:
        
        print(" For alpha:",alpha1,"& l1_ratio:",l1s)
        clf = ElasticNet(alpha=alpha1, l1_ratio = l1s)
        oof, predictions = run_kfold_model(clf, train, test, folds, features, target, n_fold=N_FOLD)

        train['FVC_pred'] = oof
        test['FVC_pred'] = predictions

        # baseline score
        train['Confidence'] = 100
        train['sigma_clipped'] = train['Confidence'].apply(lambda x: max(x, 70))
        train['diff'] = abs(train['FVC'] - train['FVC_pred'])
        train['delta'] = train['diff'].apply(lambda x: min(x, 1000))
        train['score'] = -math.sqrt(2)*train['delta']/train['sigma_clipped'] - np.log(math.sqrt(2)*train['sigma_clipped'])
        score = train['score'].mean()
        print(score)

        def loss_func(weight, row):
            confidence = weight
            sigma_clipped = max(confidence, 70)
            diff = abs(row['FVC'] - row['FVC_pred'])
            delta = min(diff, 1000)
            score = -math.sqrt(2)*delta/sigma_clipped - np.log(math.sqrt(2)*sigma_clipped)
            return -score

        results = []
        tk0 = tqdm(train.iterrows(), total=len(train))
        for _, row in tk0:
            loss_partial = partial(loss_func, row=row)
            weight = [100]
            result = sp.optimize.minimize(loss_partial, weight, method='SLSQP')
            x = result['x']
            results.append(x[0])

        # optimized score
        train['Confidence'] = results
        train['sigma_clipped'] = train['Confidence'].apply(lambda x: max(x, 70))
        train['diff'] = abs(train['FVC'] - train['FVC_pred'])
        train['delta'] = train['diff'].apply(lambda x: min(x, 1000))
        train['score'] = -math.sqrt(2)*train['delta']/train['sigma_clipped'] - np.log(math.sqrt(2)*train['sigma_clipped'])
        score = train['score'].mean()
        print(score)


# ## Predict Confidence

# In[ ]:


TARGET = 'Confidence'

target = train[TARGET]
test[TARGET] = np.nan

# features
cat_features = ['Sex', 'SmokingStatus']
num_features = [c for c in test.columns if (test.dtypes[c] != 'object') & (c not in cat_features)]
features = num_features + cat_features
drop_features = [ID, TARGET, 'predict_Week', 'base_Week', 'FVC', 'FVC_pred']
features = [c for c in features if c not in drop_features]

oof, predictions = run_kfold_model(clf, train, test, folds, features, target, n_fold=N_FOLD)


# In[ ]:


train['Confidence'] = oof
train['sigma_clipped'] = train['Confidence'].apply(lambda x: max(x, 70))
train['diff'] = abs(train['FVC'] - train['FVC_pred'])
train['delta'] = train['diff'].apply(lambda x: min(x, 1000))
train['score'] = -math.sqrt(2)*train['delta']/train['sigma_clipped'] - np.log(math.sqrt(2)*train['sigma_clipped'])
score = train['score'].mean()
print(score)


# In[ ]:


test['Confidence'] = predictions
test = test.reset_index()


# # Submission

# In[ ]:


sub = submission[['Patient_Week']].merge(test[['Patient_Week', 'FVC_pred', 'Confidence']], on='Patient_Week')
sub = sub.rename(columns={'FVC_pred': 'FVC'})

for i in range(len(otest)):
    sub.loc[sub['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'FVC'] = otest.FVC[i]
    sub.loc[sub['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'Confidence'] = 0.1
    
sub[sub.Confidence<1]

sub.to_csv('submission.csv', index=False, float_format='%.1f')

