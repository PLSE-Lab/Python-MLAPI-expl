#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import gc

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import xgboost as xgb

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

import xgboost as xgb

from IPython.display import display # Allows the use of display() for DataFrames

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Read train and test files
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[ ]:


X_train = train_df.drop(["ID", "target"], axis=1)
y_train = np.log1p(train_df["target"].values)

X_test = test_df.drop(["ID"], axis=1)


# In[ ]:


print("Train set size: {}".format(X_train.shape))
print("Test set size: {}".format(X_test.shape))


# Remove constant features(std=0)

# In[ ]:


# check and remove constant columns
colsToRemove = []
for col in X_train.columns:
    if X_train[col].std() == 0: 
        colsToRemove.append(col)
        
# remove constant columns in the training set
X_train.drop(colsToRemove, axis=1, inplace=True)

# remove constant columns in the test set
X_test.drop(colsToRemove, axis=1, inplace=True) 

print("Removed `{}` Constant Columns\n".format(len(colsToRemove)))
print(colsToRemove)


# In[ ]:


gc.collect()
print("Train set size: {}".format(X_train.shape))
print("Test set size: {}".format(X_test.shape))


# REMOVE DUPLICATE COLOMNS

# In[ ]:


get_ipython().run_cell_magic('time', '', '# The other way to drop duplicate columns is to transpose our DatFrame and use the pandas routine - drop_duplicates. \n# df.T.drop_duplicates().T. However, transposing is a bad idea when working with large DataFrames. But this option is fine in this case.\n# Check and remove duplicate columns\ncolsToRemove = []\ncolsScaned = []\ndupList = {}\n\ncolumns = X_train.columns\n\nfor i in range(len(columns)-1):\n    v = X_train[columns[i]].values\n    dupCols = []\n    for j in range(i+1,len(columns)):\n        if np.array_equal(v, X_train[columns[j]].values):\n            colsToRemove.append(columns[j])\n            if columns[j] not in colsScaned:\n                dupCols.append(columns[j]) \n                colsScaned.append(columns[j])\n                dupList[columns[i]] = dupCols\n                \n# remove duplicate columns in the training set\nX_train.drop(colsToRemove, axis=1, inplace=True) \n\n# remove duplicate columns in the testing set\nX_test.drop(colsToRemove, axis=1, inplace=True)\n\nprint("Removed `{}` Duplicate Columns\\n".format(len(dupList)))\nprint(dupList)')


# In[ ]:


def drop_sparse(train, test):
    flist = [x for x in train.columns if not x in ['ID','target']]
    for f in flist:
        if len(np.unique(train[f]))<2:
            train.drop(f, axis=1, inplace=True)
            test.drop(f, axis=1, inplace=True)
    return train, test


# In[ ]:


get_ipython().run_cell_magic('time', '', 'X_train, X_test = drop_sparse(X_train, X_test)')


# In[ ]:


gc.collect()
print("Train set size: {}".format(X_train.shape))
print("Test set size: {}".format(X_test.shape))


# In[ ]:


def add_SumZeros(train, test, features):
    flist = [x for x in train.columns if not x in ['ID','target']]
    if 'SumZeros' in features:
        train.insert(1, 'SumZeros', (train[flist] == 0).astype(int).sum(axis=1))
        test.insert(1, 'SumZeros', (test[flist] == 0).astype(int).sum(axis=1))
    flist = [x for x in train.columns if not x in ['ID','target']]

    return train, test


# In[ ]:


get_ipython().run_cell_magic('time', '', "X_train, X_test = add_SumZeros(X_train, X_test, ['SumZeros'])")


# In[ ]:


gc.collect()
print("Train set size: {}".format(X_train.shape))
print("Test set size: {}".format(X_test.shape))


# In[ ]:


def add_SumValues(train, test, features):
    flist = [x for x in train.columns if not x in ['ID','target']]
    if 'SumValues' in features:
        train.insert(1, 'SumValues', (train[flist] != 0).astype(int).sum(axis=1))
        test.insert(1, 'SumValues', (test[flist] != 0).astype(int).sum(axis=1))
    flist = [x for x in train.columns if not x in ['ID','target']]

    return train, test


# In[ ]:


get_ipython().run_cell_magic('time', '', "X_train, X_test = add_SumValues(X_train, X_test, ['SumValues'])")


# In[ ]:


gc.collect()
print("Train set size: {}".format(X_train.shape))
print("Test set size: {}".format(X_test.shape))


# In[ ]:


def add_OtherAgg(train, test, features):
    flist = [x for x in train.columns if not x in ['ID','target','SumZeros','SumValues']]
    if 'OtherAgg' in features:
        train['Mean']   = train[flist].mean(axis=1)
        train['Median'] = train[flist].median(axis=1)
        train['Mode']   = train[flist].mode(axis=1)
        train['Max']    = train[flist].max(axis=1)
        train['Var']    = train[flist].var(axis=1)
        train['Std']    = train[flist].std(axis=1)
        
        test['Mean']   = test[flist].mean(axis=1)
        test['Median'] = test[flist].median(axis=1)
        test['Mode']   = test[flist].mode(axis=1)
        test['Max']    = test[flist].max(axis=1)
        test['Var']    = test[flist].var(axis=1)
        test['Std']    = test[flist].std(axis=1)
    flist = [x for x in train.columns if not x in ['ID','target','SumZeros','SumValues']]

    return train, test


# In[ ]:


get_ipython().run_cell_magic('time', '', "X_train, X_test = add_OtherAgg(X_train, X_test, ['OtherAgg'])")


# In[ ]:


gc.collect()
print("Train set size: {}".format(X_train.shape))
print("Test set size: {}".format(X_test.shape))


# In[ ]:


from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.decomposition import TruncatedSVD, FastICA


# In[ ]:


PERC_TRESHOLD = 0.98   ### Percentage of zeros in each feature ###
N_COMP = 97            ### Number of decomposition components ###

print("tSVD")
tsvd = TruncatedSVD(n_components=N_COMP, random_state=17)
tsvd_results_train = tsvd.fit_transform(X_train)
tsvd_results_test = tsvd.transform(X_test)
print("Append decomposition components to datasets...")
for i in range(1, N_COMP + 1):
    X_train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
    X_test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]
print('\nTrain shape: {}\nTest shape: {}'.format(X_train.shape, X_test.shape))


# In[ ]:


import lightgbm as lgb


# In[ ]:


def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
        "objective" : "regression",
        "metric" : "rmse",
        'boosting_type' : 'goss',
        'max_depth' : 5,#-1
        "num_leaves" : 20,#20
        "learning_rate" : 0.01,#0.01
        #"bagging_fraction" : 0.6,#0.7 #0.8 #0.3
        "feature_fraction" : 0.6,#0.7 #0.5
        #"bagging_freq" : 2, #10 #20
        "bagging_seed" : 42, #2018
        "verbosity" : -1,
        'lambda_l2' : 0.000001,#0.1
        'lambda_l1' : 0.00001,#0,
        'max_bin' : 200 #default=250 #200 #170 #120 #90

    }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgtrain, lgval], early_stopping_rounds=100, 
                      verbose_eval=200, evals_result=evals_result)
    
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    return pred_test_y, model, evals_result


# In[ ]:


# Training LGB
#seeds = [42, 2018]
seeds = [42]
pred_test_full_seed = 0
for seed in seeds:
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=seed)
    pred_test_full = 0
    for dev_index, val_index in kf.split(X_train):
        dev_X, val_X = X_train.loc[dev_index,:], X_train.loc[val_index,:]
        dev_y, val_y = y_train[dev_index], y_train[val_index]
        pred_test, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, X_test)
        pred_test_full += pred_test
    pred_test_full /= 5.
    pred_test_full = np.expm1(pred_test_full)
    pred_test_full_seed += pred_test_full
    print("Seed {} completed....".format(seed))
pred_test_full_seed /= np.float(len(seeds))

print("LightGBM Training Completed...")


# In[ ]:


# feature importance
print("Features Importance...")
gain = model.feature_importance('gain')
featureimp = pd.DataFrame({'feature':model.feature_name(), 
                   'split':model.feature_importance('split'), 
                   'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
print(featureimp[:15])


# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')

sub_lgb = pd.DataFrame()
sub_lgb["target"] = pred_test_full_seed
sub["target"] = sub_lgb["target"]


# In[ ]:


print(sub.head())
sub.to_csv('Ensemble9.csv', index=False)


# In[ ]:


ls


# In[ ]:




