#!/usr/bin/env python
# coding: utf-8

# Kaggler's EmbeddingEncoder + TargetEncoder + AutoLGB

# In[ ]:


get_ipython().system('pip install kaggler==0.8.7')


# In[ ]:


import kaggler
print(kaggler.__version__)


# In[ ]:


import warnings
warnings.filterwarnings("ignore")

import gc
import joblib
import lightgbm as lgb
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from kaggler.metrics import auc
from kaggler.model import AutoLGB
from kaggler.preprocessing import EmbeddingEncoder, TargetEncoder, LabelEncoder


# In[ ]:


trn = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")
tst = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv")
sample = pd.read_csv("/kaggle/input/cat-in-the-dat/sample_submission.csv")


# In[ ]:


y_trn = trn['target']


# In[ ]:


for col in trn.columns:
    print('{:>8s}: {:6d}'.format(col, trn[col].nunique()))


# In[ ]:


features = [x for x in trn.columns if x not in ['id', 'target']]

features_to_emb = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_3', 'ord_4', 'ord_5']
n_emb = [16, 16, 20, 20, 30, 4, 8, 16]


# In[ ]:


features_emb = []
for n, col in zip(n_emb, features_to_emb):
    features_emb += ['{}_{}'.format(col, i + 1) for i in range(n)]


# In[ ]:


n_fold = 5
seed = 42
cv = StratifiedKFold(n_splits=n_fold, random_state=seed)


# In[ ]:


le = LabelEncoder(min_obs=50)
te = TargetEncoder(smoothing=1, min_samples=50, cv=cv)

X_trn = pd.concat([le.fit_transform(trn[features]), te.fit_transform(trn[features], y_trn)], axis=1)
X_tst = pd.concat([le.transform(tst[features]), te.transform(tst[features])], axis=1)
features = ['le_{}'.format(col) for col in features] + ['te_{}'.format(col) for col in features]

X_trn.columns = features
X_tst.columns = features
print(X_trn.shape, X_tst.shape)


# In[ ]:


p = np.zeros((trn.shape[0],))
p_tst = np.zeros((tst.shape[0],))

features += features_emb
for i, (i_trn, i_val) in enumerate(cv.split(X_trn, y_trn), 1):
    y_trn_cv = y_trn[i_trn].reset_index(drop=True)

    ee = EmbeddingEncoder(cat_cols=features_to_emb, num_cols=[], n_emb=n_emb, random_state=seed)
    X_emb_trn = ee.fit_transform(trn.loc[i_trn, features_to_emb], y_trn_cv)
    X_emb_val = ee.transform(trn.loc[i_val, features_to_emb])
    X_emb_tst = ee.transform(tst[features_to_emb])
    
    X_trn_cv = pd.concat([X_trn.loc[i_trn].reset_index(drop=True), 
                          pd.DataFrame(X_emb_trn, columns=features_emb)], axis=1)
    X_val_cv = pd.concat([X_trn.loc[i_val].reset_index(drop=True), 
                          pd.DataFrame(X_emb_val, columns=features_emb)], axis=1)
    X_tst_cv = pd.concat([X_tst, pd.DataFrame(X_emb_tst, columns=features_emb)], axis=1)
   
    if i == 1:
        print('Feature selection and parameter tuning with CV #{}'.format(i))
        model = AutoLGB(objective='binary', metric='auc', sample_size=50000, random_state=seed)
        model.tune(X_trn_cv, y_trn_cv)
        print('{} features selected out of {}'.format(len(model.features), len(features)))
        
    model.fit(X_trn_cv, y_trn_cv)
    p[i_val] = model.predict(X_val_cv)
    print('AUC (CV #{}): {:.4f}'.format(i, auc(y_trn[i_val], p[i_val])))
    p_tst += model.predict(X_tst_cv) / n_fold
    
print('AUC CV: {:.4f}'.format(auc(y_trn, p)))


# In[ ]:


print("Saving submission file")
submission = pd.DataFrame.from_dict({
    'id': tst.id.values,
    'target': p_tst
})
submission.to_csv("submission.csv", index=False)


# In[ ]:




