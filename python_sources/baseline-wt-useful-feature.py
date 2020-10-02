#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import gc
import os

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


os.listdir('../input/make-base-data')


# In[ ]:


DATA_DIR = '../input/make-base-data'
folder_path = '../input/ieee-fraud-detection/'


# In[ ]:


sub = pd.read_csv(f'{folder_path}sample_submission.csv')


# In[ ]:


train = pd.read_pickle(os.path.join(DATA_DIR,'train.pkl'))
test = pd.read_pickle(os.path.join(DATA_DIR,'test.pkl'))


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


useless_feature = ['id_16', 'V181', 'V50', 'V41', 'M1', 'id_37',
 'id_36', 'V28', 'V71', 'id_35', 'V191', 'V193', 'C3', 'V186',
 'V196', 'id_29', 'V65', 'id_28', 'V27', 'id_27', 'V68', 'V114',
 'V325', 'V31', 'V95', 'V101', 'V125', 'V21', 'V17', 'V122',
 'V120', 'V119', 'V107', 'V108', 'V176', 'V110', 'V118', 'V117',
 'V269', 'V116', 'V14', 'V305', 'V339', 'V11', 'V80', 'V84',
 'V144', 'V334', 'V88', 'V113', 'V89', 'V231', 'V1', 'V236',
 'V240', 'V241', 'V138', 'V137', 'V9', 'V177']


# In[ ]:


cols_to_drop = useless_feature

train = train.drop(cols_to_drop, axis=1)
test = test.drop(cols_to_drop, axis=1)


# In[ ]:


target = train['isFraud'].copy()

X_train = train.drop('isFraud', axis=1)
X_test = test


# In[ ]:


del train, test
gc.collect()


# In[ ]:


splits = 5
folds = StratifiedKFold(n_splits = splits, random_state=42)
oof = np.zeros(len(X_train))
predictions = np.zeros(len(X_test))

feature_importances = pd.DataFrame()
feature_importances['feature']=X_train.columns


# In[ ]:


params = {'learning_rate': 0.1,
          'metric': 'auc',
          'random_state': 42
         }


# In[ ]:


for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train.values, target.values)):
    print("Fold {}".format(fold_))
    train_df, y_train_df = X_train.iloc[trn_idx], target.iloc[trn_idx]
    valid_df, y_valid_df = X_train.iloc[val_idx], target.iloc[val_idx]
    
    
    trn_data = lgb.Dataset(train_df, label=y_train_df)
    
    val_data = lgb.Dataset(valid_df, label=y_valid_df)
    
    clf = lgb.train(params,
                    trn_data,
                    100,
                    valid_sets = [trn_data, val_data],
                    verbose_eval=50,
                    early_stopping_rounds=50)
    feature_importances[f'fold_{fold_ + 1}'] = clf.feature_importance()

    pred = clf.predict(valid_df)
    oof[val_idx] = pred
    print("auc = ", roc_auc_score(y_valid_df, pred))
    predictions += clf.predict(X_test) / splits
    


# In[ ]:


sub['isFraud'] = predictions
sub.to_csv("submission.csv", index=False)


# In[ ]:


feature_importances['average'] = feature_importances[[f'fold_{fold_n + 1}' for fold_n in range(folds.n_splits)]].mean(axis=1)
feature_importances.to_csv('feature_importances.csv')

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(16, 16))
sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature');
plt.title('50 TOP feature importance over {} folds average'.format(folds.n_splits));

