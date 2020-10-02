#!/usr/bin/env python
# coding: utf-8

# # Categorical Feature Encoding Challenge 
# Binary classification, with every feature a categorical)
# 
# In this competition, you will be predicting the probability [0, 1] of a binary target column.
# 
# The data contains binary features (bin_*), nominal features (nom_*), ordinal features (ord_*) as well as (potentially cyclical) day (of the week) and month features. The string ordinal features ord_{3-5} are lexically ordered according to string.ascii_letters.
# 
# Since the purpose of this competition is to explore various encoding strategies, the data has been simplified in that (1) there are no missing values, and (2) the test set does not contain any unseen feature values (See this). (Of course, in real-world settings both of these factors are often important to consider!)

# # Import libraries

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings('ignore')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Import Datasets

# In[ ]:


train = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv')
test  = pd.read_csv('/kaggle/input/cat-in-the-dat/test.csv')
sub   = pd.read_csv('/kaggle/input/cat-in-the-dat/sample_submission.csv')


# # Data exploration

# In[ ]:


train.shape, test.shape, sub.shape


# In[ ]:


train.head(2)


# In[ ]:


test.head(2)


# In[ ]:


sub.head(2)


# In[ ]:


bin_cols = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']
nom_cols = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']
ord_cols = ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']
cyc_cols = ['day', 'month']


# In[ ]:


df = train.append(test, ignore_index=True, sort=False)
df.head(2)


# In[ ]:


df.shape


# In[ ]:


from pandas.api.types import CategoricalDtype 

ord_1 = CategoricalDtype(categories=['Novice', 'Contributor','Expert', 'Master', 'Grandmaster'], ordered=True)
ord_2 = CategoricalDtype(categories=['Freezing', 'Cold', 'Warm', 'Hot', 'Boiling Hot', 'Lava Hot'], ordered=True)
ord_3 = CategoricalDtype(categories=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'], ordered=True)
ord_4 = CategoricalDtype(categories=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'], ordered=True)

df['ord_1'] = df['ord_1'].astype(ord_1)
df['ord_2'] = df['ord_2'].astype(ord_2)
df['ord_3'] = df['ord_3'].astype(ord_3)
df['ord_4'] = df['ord_4'].astype(ord_4)


# In[ ]:


def date_cyc_enc(df, col, max_vals):
    df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_vals)
    df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_vals)
    return df

df = date_cyc_enc(df, 'day', 7)
df = date_cyc_enc(df, 'month', 12)

#df.drop(['day','month'], axis=1, inplace=True)


# In[ ]:


df.head(1)


# In[ ]:


df.dtypes


# In[ ]:


# astype('category') and this mapping gave same results
df['bin_3'] = df['bin_3'].map({'T':1, 'F':0, 'Y':1, 'N':0})
df['bin_4'] = df['bin_4'].map({'T':1, 'F':0, 'Y':1, 'N':0})


# In[ ]:


# astype('category') gave better results than get_dummies. Commenting this code
# df = pd.get_dummies(df, columns=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'], prefix=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'], drop_first=True)


# In[ ]:


high_card_feats = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']

for col in high_card_feats:
    df[f'hash_{col}'] = df[col].apply( lambda x: hash(str(x)) % 5000 )
    
for col in high_card_feats:
    enc_nom_1 = (df.groupby(col).size()) / len(df)
    df[f'freq_{col}'] = df[col].apply(lambda x : enc_nom_1[x])


# In[ ]:


df.head(1)


# In[ ]:


for col in df.columns:
    if col != 'target' and df[col].dtype == 'O':
        df[col] = df[col].astype('category')


# In[ ]:


from sklearn.metrics import roc_auc_score


# In[ ]:


df.drop(['id'], axis=1, inplace=True)

train_df = df[df['target'].isnull()!=True]
test_df = df[df['target'].isnull()==True]
test_df.drop('target', axis=1, inplace=True)


# # Train test split

# In[ ]:


X = train_df.drop(labels=['target'], axis=1)
y = train_df['target'].values

from sklearn.model_selection import train_test_split
X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)


# # Build the model

# In[ ]:


from lightgbm import LGBMClassifier
lgbm = LGBMClassifier(boosting_type='gbdt', 
                      num_leaves=31, 
                      max_depth=-1, 
                      learning_rate=0.02, 
                      n_estimators=2000, 
                      subsample_for_bin=200000, 
                      min_child_samples=20, 
                      colsample_bytree=0.1, 
                      random_state=0)
lgbm.fit(X_train, y_train, eval_set=[(X_cv, y_cv)], eval_metric='auc', early_stopping_rounds=200, verbose=100)
y_pred = lgbm.predict_proba(X_cv)[:,-1]


# In[ ]:


roc_auc_score(y_cv, y_pred) 


# # Predict on test set

# In[ ]:


Xtest = test_df


# In[ ]:


err=[]
y_pred_tot=[]
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedStratifiedKFold
fold=StratifiedKFold(n_splits=10, shuffle=True, random_state=1994)
i=1
for train_index, test_index in fold.split(X,y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    lgbm.fit(X_train, y_train, eval_set=[(X_cv, y_cv)], eval_metric='auc', early_stopping_rounds=100, verbose=100)
    preds = lgbm.predict_proba(X_test)[:,-1]
    
    print("ROC AUC Score: ", roc_auc_score(y_test, preds))
    err.append(roc_auc_score(y_test,preds))
    p = lgbm.predict_proba(Xtest)[:,-1]
    print(f'-------------------- Fold {i} completed !!! ------------------')
    i=i+1
    y_pred_tot.append(p)


# In[ ]:


err_avg = np.mean(err,0)
err_avg


# In[ ]:


y_pred = np.mean(y_pred_tot,0)
y_pred


# In[ ]:


sub['target'] = y_pred


# In[ ]:


sub.head()


# In[ ]:


sub.to_csv('10fold_lgbm.csv', index=False)

