#!/usr/bin/env python
# coding: utf-8

# ## Kernels and discussions used in this kernel
# - [Olivier's kernel](https://www.kaggle.com/ogrellier/plasticc-in-a-kernel-meta-and-data)
# - [Alexander Firsov's kernel](https://www.kaggle.com/alexfir/fast-test-set-reading)
# - [Iprapas' kernel](https://www.kaggle.com/iprapas/ideas-from-kernels-and-discussion-lb-1-135)
# - [Chia-Ta Tsai's kernel](https://www.kaggle.com/cttsai/forked-lgbm-w-ideas-from-kernels-and-discuss)
# - [Lving's kernel](https://www.kaggle.com/qianchao/smote-with-imbalance-data)
# - [My something different kernel](https://www.kaggle.com/jimpsull/something-different)
# - [My Smote the training set kernel](https://www.kaggle.com/jimpsull/smote-the-training-sets)

# ## The purpose of this kernel is to bring together features
# - the first 69 are from our 1.080 kernel which came via Oliver, Iprapas, and Chia-ta Tsai
# - integrating smote brought that to 1.052
# - Adding my custom features improved 0.015
# - ongoing efforts and tried and failed lists at the end of the kernel
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir('../input'))
print(os.listdir("../input/writefeaturetablefromsmotedartset"))
print(os.listdir('../input/normalizesomethingdifferentfeatures'))
# Any results you write to the current directory are saved as output.


# ## From Chia-Ta Tsai's script

# In[ ]:


"""

This script is forked from chia-ta tsai's kernel of which he said:

This script is forked from iprapas's notebook 
https://www.kaggle.com/iprapas/ideas-from-kernels-and-discussion-lb-1-135

#    https://www.kaggle.com/ogrellier/plasticc-in-a-kernel-meta-and-data
#    https://www.kaggle.com/c/PLAsTiCC-2018/discussion/70908
#    https://www.kaggle.com/meaninglesslives/simple-neural-net-for-time-series-classification
#
"""

import sys, os
import argparse
import time
from datetime import datetime as dt
import gc; gc.enable()
from functools import partial, wraps

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
np.warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from tsfresh.feature_extraction import extract_features
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ## Olivier's functions

# In[ ]:



def multi_weighted_logloss(y_true, y_preds, classes, class_weights):
    """
    refactor from
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    y_p = y_preds.reshape(y_true.shape[0], len(classes), order='F')
    # Trasform y_true in dummies
    y_ohe = pd.get_dummies(y_true)
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weights[k] for k in sorted(class_weights.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss


# In[ ]:


def lgbm_multi_weighted_logloss(y_true, y_preds):
    """
    refactor from
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """  
    # Taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weights = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}

    loss = multi_weighted_logloss(y_true, y_preds, classes, class_weights)
    return 'wloss', loss, False


def xgb_multi_weighted_logloss(y_predicted, y_true, classes, class_weights):
    loss = multi_weighted_logloss(y_true.get_label(), y_predicted, 
                                  classes, class_weights)
    return 'wloss', loss


# ## Function to save feature importances (not sure who authored it)

# In[ ]:



def save_importances(importances_):
    mean_gain = importances_[['gain', 'feature']].groupby('feature').mean()
    importances_['mean_gain'] = importances_['feature'].map(mean_gain['gain'])
    return importances_


# ## This method is our biggest contribution to our current score
# - This smote method improved iprapas kernel from 1.135 --> 1.110 and Chia-Ta Tsai's from 1.080 --> 1.052
# - The biggest challeng in integrating it was the data structures (pandas DataFrames vs Numpy arrays, mixed usage of data structures)

# In[ ]:



from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import numpy as np # linear algebra
import pandas as pd

#modify to work with kfold
#def smoteAdataset(Xig, yig, test_size=0.2, random_state=0):
def smoteAdataset(Xig_train, yig_train, Xig_test, yig_test):
    
        
    sm=SMOTE(random_state=2)
    Xig_train_res, yig_train_res = sm.fit_sample(Xig_train, yig_train.ravel())

        
    return Xig_train_res, pd.Series(yig_train_res), Xig_test, pd.Series(yig_test)


# ## This is Olivier and Iprapas method but we integrated our Smote method into it

# In[ ]:



def lgbm_modeling_cross_validation(params,
                                   full_train, 
                                   y, 
                                   classes, 
                                   class_weights, 
                                   nr_fold=12, 
                                   random_state=1):

    # Compute weights
    w = y.value_counts()
    weights = {i : np.sum(w) / w[i] for i in w.index}
   # print(weights)
   # weights=class_weights
    clfs = []
    importances = pd.DataFrame()
    folds = StratifiedKFold(n_splits=nr_fold, 
                            shuffle=True, 
                            random_state=random_state)
    
    oof_preds = np.zeros((len(full_train), np.unique(y).shape[0]))
    for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
        trn_x, trn_y = full_train.iloc[trn_], y.iloc[trn_]
        val_x, val_y = full_train.iloc[val_], y.iloc[val_]
        
                
        trn_xa, trn_y, val_xa, val_y=smoteAdataset(trn_x.values, trn_y.values, val_x.values, val_y.values)
        trn_x=pd.DataFrame(data=trn_xa, columns=trn_x.columns)
    
        val_x=pd.DataFrame(data=val_xa, columns=val_x.columns)
        
        clf = LGBMClassifier(**params)
        clf.fit(
            trn_x, trn_y,
            eval_set=[(trn_x, trn_y), (val_x, val_y)],
            eval_metric=lgbm_multi_weighted_logloss,
            verbose=100,
            early_stopping_rounds=50,
            sample_weight=trn_y.map(weights)
        )
        clfs.append(clf)

        oof_preds[val_, :] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)
        print('no {}-fold loss: {}'.format(fold_ + 1, 
              multi_weighted_logloss(val_y, oof_preds[val_, :], 
                                     classes, class_weights)))
    
        imp_df = pd.DataFrame({
                'feature': full_train.columns,
                'gain': clf.feature_importances_,
                'fold': [fold_ + 1] * len(full_train.columns),
                })
        importances = pd.concat([importances, imp_df], axis=0, sort=False)

    score = multi_weighted_logloss(y_true=y, y_preds=oof_preds, 
                                   classes=classes, class_weights=class_weights)
    print('MULTI WEIGHTED LOG LOSS: {:.5f}'.format(score))
    df_importances = save_importances(importances_=importances)
    df_importances.to_csv('lgbm_importances.csv', index=False)
    
    return clfs, score


# ## These methods have several contributors
# - I'm not sure that they're still needed now that I've extracted the features from Chia-Ta Tsai's script
# - But when I tried to run the prediction on test all at once the kernel crashed
# - So I modified to read testdf in chunks and predict bit by bit

# In[ ]:



def predict_chunk(df_, clfs_, features, train_mean):
    # Group by object id    
    agg_ = df_
    # Merge with meta data
    full_test = agg_.reset_index()
    #print(full_test.head())

    full_test = full_test.fillna(0)
    full_test = full_test.round(5)
    # Make predictions
    preds_ = None
    for clf in clfs_:
        if preds_ is None:
            preds_ = clf.predict_proba(full_test[features]) / len(clfs_)
        else:
            preds_ += clf.predict_proba(full_test[features]) / len(clfs_)
            
    #going to recalc 99 below anyways
    # Compute preds_99 as the proba of class not being any of the others
    # preds_99 = 0.1 gives 1.769
    preds_99 = np.ones(preds_.shape[0])
    
    
    for i in range(preds_.shape[1]):
        preds_99 *= (1 - preds_[:, i])

    # Create DataFrame from predictions
    preds_df_ = pd.DataFrame(preds_, columns=['class_' + str(s) for s in clfs_[0].classes_])
    preds_df_['object_id'] = full_test['object_id']
    preds_df_['class_99'] = 0.14 * preds_99 / np.mean(preds_99) 

    return preds_df_


# ## Remove the end effect with good chunksize choice
# - testdf.shape[0]%40615=0 so there's no special 'end case'

# In[ ]:


def process_test(clfs, 
                 testdf,
                 full_train,
                 train_mean,
                 filename='submission.csv',
                 chunks=40615):

    import time

    start = time.time()
    chunks = 40615
    testdf=testdf.round(5)
    
    testdf.to_csv(filename, index=False)
    for i_c, df in enumerate(pd.read_csv(filename, chunksize=chunks, iterator=True)):

        print(df.shape)
        preds_df = predict_chunk(df_=df,
                                 clfs_=clfs,
                                 features=full_train.columns,
                                 train_mean=train_mean)

        if i_c == 0:
            preds_df.to_csv('predictions.csv', header=True, mode='a', index=False)
        else:
            preds_df.to_csv('predictions.csv', header=False, mode='a', index=False)

        del preds_df
        gc.collect()

        print('%15d done in %5.1f minutes' % (chunks * (i_c + 1), (time.time() - start) / 60), flush=True)

    return


# ## Surprisingly making changes to these parameters didn't have a big impact on score
# - I thought adding Smote since they optimized would leave room for re-optimization
# - But couldn't get scores to come up 

# In[ ]:


best_params = {
            'device': 'cpu', 
            'objective': 'multiclass', 
            'num_class': 14, 
            'boosting_type': 'gbdt', 
            'n_jobs': -1, 
            'max_depth': 6, 
            'n_estimators': 1000, 
            'subsample_freq': 2, 
            'subsample_for_bin': 5000, 
            'min_data_per_group': 100, 
            'max_cat_to_onehot': 4, 
            'cat_l2': 1.0, 
            'cat_smooth': 59.5, 
            'max_cat_threshold': 32, 
            'metric_freq': 10, 
            'verbosity': -1, 
            'metric': 'multi_logloss', 
            'xgboost_dart_mode': False, 
            'uniform_drop': False, 
            'colsample_bytree': 0.5, 
            'drop_rate': 0.173, 
            'learning_rate': 0.025, 
            'max_drop': 5, 
            'min_child_samples': 10, 
            'min_child_weight': 200.0, 
            'min_split_gain': 0.01, 
            'num_leaves': 7, 
            'reg_alpha': 0.1, 
            'reg_lambda': 0.00023, 
            'skip_drop': 0.44, 
            'subsample': 0.75}


# ## Load and merge the training data
# - trainingDartDf is from Chai-Ta Tsai's kernel
# - trainingJimsDf is from my somethingDifferent kernel
# 

# In[ ]:


#Here is a change from the script
#training features
trainingDartDf=pd.read_csv('../input/writefeaturetablefromsmotedartset/trainingFeatures1039.csv')
trainingJimsDf=pd.read_csv('../input/normalizesomethingdifferentfeatures/traindfNormal.csv')
if 'Unnamed: 0' in trainingDartDf.columns:
    trainingDartDf=trainingDartDf.drop('Unnamed: 0', axis=1)
print(trainingDartDf.shape)
#trainingDartDf.head()
columnsToAdd=['outlierScore', 'hipd', 'lipd', 'highEnergy_transitory_1.0_TF',
          'highEnergy_transitory_1.5_TF', 'lowEnergy_transitory_1.0_TF', 
          'lowEnergy_transitory_1.5_TF']

for column in columnsToAdd:
    trainingDartDf.loc[:,column]=trainingJimsDf.loc[:,column]

traindf=trainingDartDf

#from the 1.052 kernel
del traindf['hostgal_specz']
del traindf['ra'], traindf['decl'], traindf['gal_l'], traindf['gal_b']
del traindf['ddf']


print(traindf.shape)
traindf.head()


# ## Load the test data
# - be careful with memory
# - if you bring together features from multiple sources it is best to delete the dataframes once you have the features you need

# In[ ]:


#test features
testDartDf=pd.read_csv('../input/writefeaturetablefromsmotedartset/feat_0.648970_2018-11-23-09-00.csv')
testJimsDf=pd.read_csv('../input/normalizesomethingdifferentfeatures/testdfNormal.csv')

if 'Unnamed: 0' in testDartDf.columns:
    testDartDf=testDartDf.drop('Unnamed: 0', axis=1)
print(testDartDf.shape)
testDartDf.head()

for column in columnsToAdd:
    testDartDf.loc[:,column]=testJimsDf.loc[:,column]

testdf=testDartDf

#from the 1.052 kernel
del testdf['hostgal_specz']
del testdf['ra'], testdf['decl'], testdf['gal_l'], testdf['gal_b']
del testdf['ddf']

testdf.shape


# ## Prep training data for Olivier & company's cross validation methods

# In[ ]:


full_train=traindf
if 'target' in full_train:
    y = full_train['target']
    del full_train['target']

classes = sorted(y.unique())    
# Taken from Giba's topic : https://www.kaggle.com/titericz
# https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
# with Kyle Boone's post https://www.kaggle.com/kyleboone
class_weights = {c: 1 for c in classes}
class_weights.update({c:2 for c in [64, 15]})
print('Unique classes : {}, {}'.format(len(classes), classes))
print(class_weights)


# ## Continue prepping traindf for cross validation, save object_ids

# In[ ]:



if 'object_id' in full_train:
    oof_df = full_train[['object_id']]
    del full_train['object_id'] 
    #del full_train['distmod'] 

train_mean = full_train.mean(axis=0)
#train_mean.to_hdf('train_data.hdf5', 'data')
pd.set_option('display.max_rows', 500)
#print(full_train.describe().T)
#import pdb; pdb.set_trace()
full_train.fillna(0, inplace=True)
print(full_train.shape)
full_train.head()


# ## The first two lines (or lack thereof) have caused me more headache than I can count
# - it has to do with numpy data types when native data types are expected

# In[ ]:


for cindex in full_train.columns:
    full_train.loc[:,cindex]=np.float64(full_train.loc[:,cindex])

eval_func = partial(lgbm_modeling_cross_validation, 
                        full_train=full_train, 
                        y=y, 
                        classes=classes, 
                        class_weights=class_weights, 
                        nr_fold=12, 
                        random_state=1)

best_params.update({'n_estimators': 1000})
    
    # modeling from CV
clfs, score = eval_func(best_params)


# ## Chai-Ta Tsai's naming convention
# - stores the CV score and the timestamp in the filename

# In[ ]:



filename = 'subm_{:.6f}_{}.csv'.format(score, 
                 dt.now().strftime('%Y-%m-%d-%H-%M'))
print('save to {}'.format(filename))
# TEST


process_test(clfs, 
             testdf,
             full_train,
             train_mean=train_mean, 
             filename=filename,
             chunks=40615)


pdf = pd.read_csv('predictions.csv')
pdf.to_csv(filename, index=False)

