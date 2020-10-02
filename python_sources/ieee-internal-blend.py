#!/usr/bin/env python
# coding: utf-8

# # Summary
# 
# I'll try to make a small summary for this blend baseline:
# 
# Step: 0. EDA (missing kernel here, I'll post later)
# 
# 
# Step: 1. Minify Data 
# > https://www.kaggle.com/kyakovlev/ieee-data-minification
# 
# 
# Step: 2. Make ground baseline with no fe:
# > https://www.kaggle.com/kyakovlev/ieee-ground-baseline and 
# > https://www.kaggle.com/kyakovlev/ieee-ground-baseline-deeper-learning
# 
# 
# Step: 3. Make a small FE and see I you can understand data you have
# >  https://www.kaggle.com/kyakovlev/ieee-ground-baseline-make-amount-useful-again and
# >  https://www.kaggle.com/kyakovlev/ieee-gb-2-make-amount-useful-again
# 
# 
# Step: 4. Find good CV strategy 
# >  https://www.kaggle.com/kyakovlev/ieee-cv-options
# and same with gap to compare results (gap in values is what we have in test set)
# https://www.kaggle.com/kyakovlev/ieee-cv-options-with-gap
# 
# Step: 4(1). Groupkfold (by timeblocks) application
# > https://www.kaggle.com/kyakovlev/ieee-lgbm-with-groupkfold-cv
# 
# 
# Step: 5. Try different set of features
# >  https://www.kaggle.com/kyakovlev/ieee-experimental
# 
# 
# Step: 6. Make deeper FE (brute force option)
# > https://www.kaggle.com/kyakovlev/ieee-fe-with-some-eda
# 
# 
# Step: 7. Features selection (missing kernel here, I'll post later)
# 
# 
# Step: 8. Hyperopt (missing kernel here, I'll post later)
# 
# 
# Step: 9. Try other models (XGBoost, CatBoost, NN - missing kernel here, I'll post later)
# > CatBoost (with categorical transformations)  https://www.kaggle.com/kyakovlev/ieee-catboost-baseline-with-groupkfold-cv
# 
# Step: 10. Try blending and stacking (missing kernel here, I'll post later)
# 
# ---
# 
# (Utils)
# 
# Some tricks that where used in fe kernel
# > https://www.kaggle.com/kyakovlev/ieee-small-tricks
# 
# Part of EDA (Just few things)
# > https://www.kaggle.com/kyakovlev/ieee-check-noise and https://www.kaggle.com/kyakovlev/ieee-simple-eda
# 
# ---
# 
# https://www.kaggle.com/c/ieee-fraud-detection/discussion/104142

# In[ ]:


# General imports
import pandas as pd
import os, sys, gc, warnings

warnings.filterwarnings('ignore')


# In[ ]:


########################### DATA LOAD/MIX/EXPORT
#################################################################################
# Simple lgbm (0.0948)
sub_1 = pd.read_csv('../input/ieee-simple-lgbm/submission.csv')

# Blend of two kernels with old features (0.9468)
sub_2 = pd.read_csv('../input/ieee-cv-options/submission.csv')

# Add new features lgbm with CV (0.09485)
sub_3 = pd.read_csv('../input/ieee-lgbm-with-groupkfold-cv/submission.csv')

# Add catboost (0.09407)
sub_4 = pd.read_csv('../input/ieee-catboost-baseline-with-groupkfold-cv/submission.csv')

# Add catboost (0.09523)
sub_5 = pd.read_csv('../input/mysub18/simple_ensemble30.csv', index_col='TransactionID')
sub_5.to_csv('submission_old.csv')
sub_6 = pd.read_csv('../input/mysub18/IEEE_add_917_9547.csv', index_col='TransactionID')
sub_7 = pd.read_csv('../input/mysub18/IEEE_add_918_9548.csv', index_col='TransactionID')
sub_7 = pd.read_csv('../input/mysub18/IEEE_version1_version2_kernel_fix.csv', index_col='TransactionID')
# sub_7['isFraud'] += sub_6['isFraud']
sub_7['isFraud'] += sub_5['isFraud'] * 1.5
print(sub_7)
sub_7.to_csv('submission.csv')

sf1 = pd.read_csv('../input/mysub18/fe_test2.csv', index_col='TransactionID')
sub_7 = sub_7.merge(sf1,on='TransactionID',how='left')
sub_7['isFraud'] = sub_7.groupby('ukey')['isFraud'].transform('mean')
sub_7[['isFraud']].to_csv('submission_proc.csv')


# In[ ]:


import numpy as np

train_transaction = pd.read_csv('../input//ieee-fraud-detection/train_transaction.csv', index_col='TransactionID')



train_f5 = pd.read_csv('../input/mysub18/fi_train4.csv', index_col='TransactionID')
train_transaction = train_transaction.merge(train_f5, how='left', left_index=True, right_index=True)
#
debug = False
if not debug:
    test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv', index_col='TransactionID')
    test_f5 = pd.read_csv('../input/mysub18/fi_test4.csv', index_col='TransactionID')
    test_transaction = test_transaction.merge(test_f5, how='left', left_index=True, right_index=True)
    test_transaction['isFruad'] = 0.0

sub = pd.read_csv('./submission.csv', index_col='TransactionID')
train_transaction['pred'] = 0.0
test_transaction['pred'] = sub['isFraud']


key = 'card123456_add1_D15_series'
key2 = 'card123456_add1_D2_series'
key3 = 'card123456_add1_D11_series'




if debug:
    cache = train_transaction[[key,key2,key3,'isFraud','pred']].values
    cache2 = train_transaction['pred'].values
else:
    cache = train_transaction.append(test_transaction)[[key, key2,key3, 'isFraud','pred']].values
    cache2 = train_transaction.append(test_transaction)['pred'].values


# In[ ]:


count00 = 0
count01 = 0
count10 = 0
count11 = 0

ukey_dict = {}
ukey2_dict = {}
ukey3_dict = {}
if debug:
    train_len = train_transaction.shape[0] * 4//5
else:
    train_len = train_transaction.shape[0]

pred_np = []
for i in range(cache.shape[0]):
    ukey = cache[i,0]
    ukey2 = cache[i, 1]
    ukey3 = cache[i, 2]
    isFraud = cache[i,3]
    pred = cache[i,4]
    ukey_dict[ukey] = ukey_dict.get(ukey,[])
    ukey2_dict[ukey2] = ukey2_dict.get(ukey2, [])
    ukey3_dict[ukey3] = ukey3_dict.get(ukey3, [])


    res = cache2[i]

    if i >= train_len and i < train_len + 800000:
        pred = 0.5

        offset = i - train_len

        if int(np.sum(ukey_dict[ukey]) > 0.5 * len(ukey_dict[ukey])) + int(np.sum(ukey2_dict[ukey2]) > 0.5 * len(ukey2_dict[ukey2])) + int(np.sum(ukey3_dict[ukey3]) > 0.5 * len(ukey3_dict[ukey3])) >= 2:
            # pred = 1
            count01 +=1
            res += 0.25
        elif int(np.sum(ukey_dict[ukey]) > 0.5 * len(ukey_dict[ukey])) + int(np.sum(ukey2_dict[ukey2]) > 0.5 * len(ukey2_dict[ukey2])) + int(np.sum(ukey3_dict[ukey3]) > 0.5 * len(ukey3_dict[ukey3])) >= 1:
            pred = 1
            count10 += 1
            res += 0.10

        elif int(np.sum(ukey_dict[ukey]) < max(0,0.2 * len(ukey_dict[ukey]))) + int(np.sum(ukey2_dict[ukey2]) < 0.2 * max(0,len(ukey2_dict[ukey2]))) + int(np.sum(ukey3_dict[ukey3]) < 0.2 * max(0,len(ukey3_dict[ukey3]))) > 2:
            pred = 0
            count11 += 1
            res *= 0.2

        if debug:
            if pred == 0 and isFraud == 0:
                count00 +=1
            elif pred == 0 and isFraud == 1:
                count01 +=1
            elif pred == 1 and isFraud == 0:
                count10 +=1
            elif pred == 1 and isFraud == 1:
                count11 +=1

    if debug:
        pred_np.append(res)
    elif i >= train_len:
        pred_np.append(res)
    if i >= train_len:
        continue

    ukey_dict[ukey].append(isFraud)
    ukey2_dict[ukey2].append(isFraud)
    ukey3_dict[ukey3].append(isFraud)

print(count00,count01,count10,count11)
if debug:
    train_transaction['pred'] = np.array(pred_np)
    from sklearn.metrics import roc_auc_score
    split_pos = train_transaction.shape[0]*4//5
    df = train_transaction.iloc[split_pos:,:].copy()
    y_test = df['isFraud']
    y_pred = df['pred']
    print(roc_auc_score(y_test,y_pred))
else:
    print(sub['isFraud'])
    print( np.array(pred_np))
    sub['isFraud'] = np.array(pred_np)
    sub.to_csv('./sub_group_d2d15d11.csv')
    

