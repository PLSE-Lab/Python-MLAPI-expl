#!/usr/bin/env python
# coding: utf-8

# Kaggler's EmbeddingEncoder (inspired by Abhishek's) + AutoLGB

# In[ ]:


get_ipython().system('pip install -U kaggler==0.8.6')


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
from kaggler.preprocessing import EmbeddingEncoder, LabelEncoder


# In[ ]:


train = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")
test = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv")
sample = pd.read_csv("/kaggle/input/cat-in-the-dat/sample_submission.csv")


# In[ ]:


for col in train.columns:
    print('{:>8s}: {:6d}'.format(col, train[col].nunique()))


# In[ ]:


features_to_emb = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_3', 'ord_4', 'ord_5']
n_emb = [16, 16, 20, 20, 30, 4, 8, 16]

features_not_to_emb = [x for x in train.columns if x not in features_to_emb + ['id', 'target']]


# In[ ]:


n_fold = 5
seed = 42
cv = StratifiedKFold(n_splits=n_fold, random_state=seed)


# In[ ]:


ee = EmbeddingEncoder(cat_cols=features_to_emb, num_cols=[], n_emb=n_emb, random_state=seed)
X_emb_trn = ee.fit_transform(train[features_to_emb], train['target'])
X_emb_tst = ee.transform(test[features_to_emb])


# In[ ]:


features_emb = []
for n, col in zip(n_emb, features_to_emb):
    features_emb += ['{}_{}'.format(col, i + 1) for i in range(n)]


# In[ ]:


lbe = LabelEncoder(min_obs=10)
train.loc[:, features_not_to_emb] = lbe.fit_transform(train[features_not_to_emb])
test.loc[:, features_not_to_emb] = lbe.transform(test[features_not_to_emb])


# In[ ]:


X_trn = pd.concat([train[features_not_to_emb], pd.DataFrame(X_emb_trn, columns=features_emb)], axis=1)
y_trn = train['target']
X_tst = pd.concat([test[features_not_to_emb], pd.DataFrame(X_emb_tst, columns=features_emb)], axis=1)
features = features_not_to_emb + features_emb


# In[ ]:


model = AutoLGB(objective='binary', metric='auc', sample_size=50000, random_state=seed)
model.tune(X_trn, y_trn)
print('{} features selected out of {}'.format(len(model.features), len(features)))


# In[ ]:


p = np.zeros((X_trn.shape[0],))
p_tst = np.zeros((X_tst.shape[0],))
for i, (i_trn, i_val) in enumerate(cv.split(X_trn, y_trn), 1):
    model.fit(X_trn.loc[i_trn, features], y_trn[i_trn])
    p[i_val] = model.predict(X_trn.loc[i_val, features])
    print('AUC (CV #{}): {:.4f}'.format(i, auc(y_trn[i_val], p[i_val])))
    p_tst += model.predict(X_tst[features]) / n_fold
    
print('AUC CV: {:.4f}'.format(auc(y_trn, p)))


# In[ ]:


print("Saving submission file")
submission = pd.DataFrame.from_dict({
    'id': test.id.values,
    'target': p_tst
})
submission.to_csv("submission.csv", index=False)

