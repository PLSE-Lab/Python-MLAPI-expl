#!/usr/bin/env python
# coding: utf-8

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


import matplotlib.pyplot as plt


# In[ ]:


input_dir = '../input'
sample = pd.read_csv(os.path.join(input_dir, 'sampleSubmission.csv'))


# In[ ]:


sample.Id.nunique()


# In[ ]:


df = pd.read_csv(os.path.join(input_dir, 'train.csv'))
dft = pd.read_csv(os.path.join(input_dir, 'test.csv'))


# In[ ]:


# target is discrete binary
df.True_y.unique()


# In[ ]:


df.isna().any()


# In[ ]:


dft.isna().any()


# In[ ]:


#lets see inferred types vs csv content
df.select_dtypes(np.number).columns


# In[ ]:


df.select_dtypes(exclude=np.number).columns


# In[ ]:


get_ipython().system("head -n 5 '../input/train.csv'")


# In[ ]:


def values_info(df, dft, col, norm=True):
    print( np.sort(df[col].unique()) )
    print(np.sort(dft[col].unique()) )
    print(df[col].value_counts(normalize=norm))
    print(dft[col].value_counts(normalize=norm))


# In[ ]:


def two_histograms(df,dft, col):
    fsize=(14,6)
    fig,ax = plt.subplots(ncols=2, figsize=fsize)
    df[col].hist(ax=ax[0])
    dft[col].hist(ax=ax[1])


# In[ ]:


two_histograms(df, dft, 'age')


# In[ ]:


values_info(df, dft, 'job')


# In[ ]:


values_info(df, dft, 'marital')


# In[ ]:


values_info(df, dft, 'education')


# In[ ]:


values_info(df, dft, 'default')


# In[ ]:


two_histograms(df, dft, 'balance')


# In[ ]:


values_info(df, dft, 'balance')


# In[ ]:


values_info(df, dft, 'housing')


# In[ ]:


values_info(df, dft, 'loan')


# In[ ]:


values_info(df, dft, 'contact')


# In[ ]:


two_histograms(df, dft, 'day')


# In[ ]:


values_info(df, dft, 'month')


# In[ ]:


two_histograms(df, dft, 'duration')


# In[ ]:


df[df.duration<60].True_y.mean()


# In[ ]:


df[df.duration>=60].True_y.mean()


# In[ ]:


df[df.duration>=240].True_y.mean()


# In[ ]:


values_info(df, dft, 'campaign')


# In[ ]:


two_histograms(df, dft, 'campaign')


# In[ ]:


two_histograms(df, dft, 'pdays')


# In[ ]:


two_histograms(df[df.previous<30], dft, 'previous')
#ok here we have some big difference but only if we get >= 30 from train


# In[ ]:


values_info(df, dft, 'previous')


# In[ ]:


values_info(df, dft, 'poutcome')


# Cols meaning:
# * 'age' 
# * 'job' type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
# * 'marital'  marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
# * 'education' categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
# * 'default'  has credit in default? (categorical: 'no','yes','unknown')
# * 'balance' 
# * 'housing'  has housing loan? (categorical: 'no','yes','unknown')
# * 'loan' has personal loan? (categorical: 'no','yes','unknown')
# * 'contact' contact communication type (categorical: 'cellular','telephone') 
# * 'day' day of contact(originally it was day of week)
# * 'month' last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
# * 'duration'  last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# * 'campaign'  number of contacts performed during this campaign and for this client (numeric, includes last contact)
# * 'pdays'  number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
# * 'previous' number of contacts performed before this campaign and for this client (numeric)
# * 'poutcome'  outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')

# In[ ]:


df.groupby("duration").agg({"True_y": ["mean", "count"]})


# In[ ]:


df.groupby("poutcome").agg({"True_y": ["mean", "count"]})


# In[ ]:


# poutcome == uknown probably is pdays == -1


# In[ ]:


df.groupby("campaign").agg({"True_y": ["mean", "count"]})


# In[ ]:


df.groupby("pdays").agg({"True_y": ["mean", "count"]})


# In[ ]:


import numpy as np
from numba import jit

@jit
def fast_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc


@jit
def fast_auc_weight(y_true, y_prob, w):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        w_i = w[i]
        nfalse += (1 - y_i) * w_i
        auc += y_i * nfalse * w_i
    auc /= (nfalse * (n - nfalse))
    return auc


def eval_auc(preds, dtrain):
    labels = dtrain.get_label()
    weights = dtrain.get_weight()
    if weights is None:
        return 'auc', fast_auc(labels, preds), True
    else:
        return 'auc', fast_auc_weight(labels, preds, weights), True


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tqdm import tqdm
import lightgbm as lgb
from sklearn.metrics import roc_auc_score


# In[ ]:


#cols = ['age', 'balance', 'duration']
num_cols = ['age', 'balance', 'duration', 'day', 'campaign', 'pdays', 'previous']
categoricals = ['default']
cols = num_cols + categoricals
train_cols = cols


# In[ ]:


target = 'True_y'
X = df[cols].copy()
y = df[target]


# In[ ]:


#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=666)
folds_count = 11
kf = KFold(n_splits=folds_count)

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': 0,  # > 1 is debug
    'boost_from_average': False,
    'boosting_type': 'gbdt',
    'feature_fraction': 1.0
}


# In[ ]:


Xsub = dft[cols].copy()
test_preds = dft.copy()
test_preds = test_preds.drop(dft.columns, axis=1)


# In[ ]:


#TODO: factorize categoricals
for c in categoricals:
    labels, uniques = pd.factorize(X[c])
    X[c] = labels
    Xsub[c] = Xsub[c].map(lambda x: uniques.get_loc(x))


# In[ ]:


#for fold, (train_index, test_index) in enumerate([(X_train.index, X_val.index)]):
for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
    print(f"fold {fold}")

    xt = X.loc[train_index]
    yt = y[train_index]
    xval = X.loc[test_index]
    yval = y[test_index]

    xt = xt[train_cols]
    xval = xval[train_cols]

    print(f"xt shape {xt.shape} ")
    print(f"xval shape {xval.shape} ")

    val_probs_cat = 0
    sub_probs_cat = 0
    val_probs_lgb = 0
    sub_probs_lgb = 0

    N = 1
    for i in range(N):
        d_train = lgb.Dataset(xt, label=yt, categorical_feature=categoricals)
        lgb_eval = lgb.Dataset(xval, label=yval, reference=d_train, categorical_feature=categoricals)

        lgb_clf = lgb.train(lgb_params,
                                d_train,
                                num_boost_round=5000,
                                valid_sets=lgb_eval,
                                early_stopping_rounds=100,
                                verbose_eval=100,
                                feval=eval_auc)

        vp = lgb_clf.predict(xval)
        val_score = roc_auc_score(yval, vp)
        train_score = roc_auc_score(yt, lgb_clf.predict(xt))
        print(f"val_score={val_score}")
        print(f"train_score={train_score}")
        
        print(f"yt mean={yt.mean()}")
        print(f"yval mean={yval.mean()}")

        val_probs_lgb += vp
        sub_probs_lgb += lgb_clf.predict(Xsub[train_cols])

    test_preds[f'fold{fold}'] = sub_probs_lgb / N


# In[ ]:


test_preds = test_preds.reset_index(drop=False)


# In[ ]:


test_preds['Id'] = test_preds['index']
fold_cols = [f'fold{n}' for n in np.arange(0,folds_count)]
test_preds['Predicted'] = np.mean(test_preds[fold_cols], axis=1)


# In[ ]:


test_preds = test_preds.drop(fold_cols + ['index'], axis=1)


# In[ ]:


test_preds.to_csv("lgb_07052019_11fold_1.csv", index=False)


# In[ ]:


get_ipython().system('ls .')


# In[ ]:




