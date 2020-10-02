#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gc
import glob
import os
import time

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import KFold, StratifiedKFold

gc.enable()


# ### Note:
# 
# This kernel is prepared for educational purposes to teach students on how to approach Kaggle and Machine Learning projects in general.
# 
# Also, it uses some elements from other kernels like NaN replacement part or GBM parameters. Thanks to the authors of those!

# ## Machine Learning projects workflow - reusable functions
# 
# ### General:
# 
# This notebook will be revolving around structuring machine learning workflow.
# Writing your code in a way which makes it reusable does have a lot of advantages.
# Functions written in this notebook are generally simple but enable easy change of parameters without change of output structure, so one can for example compute a few aggregates for a selected DataFrame and then concatenate all of them to a final training/test DF.
# 
# Sometimes it is better to put a bit more time into structuring your functions properly, making them reusable in different projects. This way, one can build a code base for different types of ML tasks and often simply copy a set of functions to use them in a new project. Of course, not always everything can be used when copied as is but often a few slight changes or improvements will make code usable again. 
# 
# ### Home Credit:
# 
# In Home Credit, some of the tables have similar structure and therefore the same function can be used on them for proper processing. There are exceptions though - Bureau Balance and Bureau require a different approach, where we first create aggregates for Bureau Balance using `SK_ID_BUREAU`, as this column is also present in Bureau, so Bureau Balance aggregates can be merged using this particular column. This is why in Bureau processing function one should account for merging the Bureau Balance onto Bureau first and then aggregating the Bureau itself. Of course, this process could be divided into two steps and two functions:
# 
# 1. merge Bureau Balance onto Bureau
# 2. aggregate Bureau
# 
# but for educational purpose the first approach was chosen.
# 
# Another thing to keep in mind is the fact that even though table structure may be the same, each DF may require a bit different approach to column processing, NaN imputation or dropping specific columns, which the model should not be trained on (such as `SK_ID_PREV`, `SK_ID_CURR` - in general index or ID columns).
# 
# In each of the processing/preparation functions emphasis is also put on column renaming. Even though at first this may seem unnecessary, as the model can be trained on columns named `['1', '2', '3'...]`, checking feature importance or performing further feature selection in an informative way may prove impossible in such case. One more thing to keep in mind is the fact that if one wants to train XGB/LGBM model on DataFrame, column names must be unique. That's one more reason to take care of this when preparing features for the model.
# 
# KFold training function has been prepared in a way to enable both easy ensembling (through simple model parameters or splits changes) and easy selection of each fold test predictions. It is much easier to simply take a mean over column axis when having predictions for each fold in a separate column than to try recovering them for already averaged set ;).

# ### 0. Parameters setting
# 
# Even though some of the parameters can hardcoded and in many cases this will not make much difference, you never really know when you may want to use this parameter again, for example to keep files placement or model parameters consistent between functions. 
# 
# This is the reason why it's better to have most of them structured as variables or even dictionaries (in cases where there are a lot of parameters).
# 
# Here, we will just set the data directory one.

# In[ ]:


data_src = '../input/'


# ### 1. Loading the data

# In[ ]:


def load_data(data_src):
    
    start_time = time.time()
    
    """
    Implement a function loading all DataFrames from the dataset.
    """
    
    print('Time it took to load all the data: {:.4f}s'.format(time.time() - start_time))
    
    return train, test, bureau, bureau_bal, prev, cred_card_bal, pos_cash_bal, ins


# In[ ]:


def load_data(data_src):
    
    start_time = time.time()
    
    train = pd.read_csv('{}application_train.csv'.format(data_src)) 
    test = pd.read_csv('{}application_test.csv'.format(data_src))
    print('Train and test tables loaded.')
    
    bureau = pd.read_csv('{}bureau.csv'.format(data_src))
    bureau_bal = pd.read_csv('{}bureau_balance.csv'.format(data_src))
    print('Bureau data loaded.')
    
    prev = pd.read_csv('{}previous_application.csv'.format(data_src))
    print('Previous applications data loaded.')
    
    cred_card_bal = pd.read_csv('{}credit_card_balance.csv'.format(data_src))
    print('Credit card balance loaded.')
    
    pos_cash_bal = pd.read_csv('{}POS_CASH_balance.csv'.format(data_src))
    print('POS cash balance loaded.')
    
    ins = pd.read_csv('{}installments_payments.csv'.format(data_src))
    print('Installments data loaded.')
    
    print('Time it took to load all the data: {:.4f}s'.format(time.time() - start_time))
    
    return train, test, bureau, bureau_bal, prev, cred_card_bal, pos_cash_bal, ins


train, test, bureau, bureau_bal, prev, cred_card_bal, pos_cash_bal, ins = load_data(data_src)


# ### 1.2. Drop 'SK_ID_PREV' column and replace values which do not make sense with NaN.

# In[ ]:


ins = ins.drop(['SK_ID_PREV'], axis=1)
prev = prev.drop(['SK_ID_PREV'], axis=1)
cred_card_bal = cred_card_bal.drop(['SK_ID_PREV'], axis=1)
pos_cash_bal = pos_cash_bal.drop(['SK_ID_PREV'], axis=1)


prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)


# ### 2.1. Aggregate bureau_balance

# In[ ]:


def prepare_bureau_bal(bureau_bal, group_column='SK_ID_BUREAU', aggregate='count'):
    
    """
    Implement a function to group bureau_bal DF in a way to enable it's merge onto bureau DF.
    Count aggregate is fine.
    Remember to reset the index in order to enable proper merge on column name.
    Remember to rename the columns but not the one the DF is grouped by (the first one), 
    as it has to retain it's original name for merge to be possible.
    It's good to have aggregate name in the columns, as you can then use the function with different aggregates
    and merge onto final DF easily, without having to deal with duplicate column names.
    Hint: this can be done if you select only columns not containing group_column string in their names.
    """
    
    return bureau_bal_group


# In[ ]:


def prepare_bureau_bal(bureau_bal, group_column='SK_ID_BUREAU', aggregate='count'):
    
    bureau_bal_prepared = bureau_bal.groupby([group_column]).agg(aggregate).reset_index()
    bureau_bal_prepared.columns = [
        'bureau_bal_{}_{}'.format(x, aggregate) if group_column not in x else x for x in bureau_bal_prepared.columns]
    
    return bureau_bal_prepared


bureau_bal_prepared = prepare_bureau_bal(bureau_bal, aggregate='count')
bureau_bal_prepared


# ### 2.2. Aggregate bureau and merge bureau_bal onto it

# In[ ]:


def prepare_bureau(bureau, bureau_bal_prepared, group_column='SK_ID_CURR', aggregate='mean'):
    
    """
    Implement a function to group bureau DF in a way to enable it's merge onto train/test DF.
    Mean aggregate is fine.
    Remember to reset the index in order to enable proper merge on column name.
    Remember to rename the columns but not the one the DF is grouped by (the first one), 
    as it has to retain it's original name for merge to be possible.
    It's good to have aggregate name in the columns, as you can then use the function with different aggregates
    and merge onto final DF easily, without having to deal with duplicate column names.
    Hint: this can be done if you select only columns not containing group_column string in their names.
    """
    
    return bureau_prepared


# In[ ]:


def prepare_bureau(bureau, bureau_bal_prepared, group_column='SK_ID_CURR', aggregate='mean'):
    
    bureau_prepared = bureau.merge(bureau_bal_prepared, how='left', on='SK_ID_BUREAU', copy=False)
    bureau_prepared = bureau_prepared.groupby([group_column]).agg(aggregate).reset_index()
    bureau_prepared.columns = [
        'bureau_{}_{}'.format(x, aggregate) if group_column not in x else x for x in bureau_prepared.columns]
    
    return bureau_prepared

bureau_prepared = prepare_bureau(bureau, bureau_bal_prepared, aggregate='mean')
bureau_prepared


# ### 2.3. Aggregate other DFs

# In[ ]:


def prepare_standard(df, group_column='SK_ID_CURR', aggregate='mean', df_name='prev'):
    
    """
    Implement a function to group other DFs in a way to enable it's merge onto train/test DF.
    This one will be very similar to prepare_bureau except for the fact that no merge of 
    bureau_balance is needed in the function body.
    One thing that is worth adding is the df_name parameter, which will be used for naming
    of columns in resulting DF.
    """
    
    return df_prepared


# In[ ]:


def prepare_standard(df, group_column='SK_ID_CURR', aggregate='mean', df_name='prev'):
    
    df_prepared = df.groupby([group_column]).agg(aggregate).reset_index()
    df_prepared.columns = [
        '{}_{}_{}'.format(df_name, x, aggregate) if group_column not in x else x for x in df_prepared.columns]
    
    return df_prepared

prev_prepared = prepare_standard(prev, aggregate='mean', df_name='prev')
prev_prepared


# In[ ]:


ins_prepared = prepare_standard(ins, aggregate='mean', df_name='ins')
cred_card_bal_prepared = prepare_standard(cred_card_bal, aggregate='mean', df_name='cred_card_bal')
pos_cash_bal_prepared = prepare_standard(pos_cash_bal, aggregate='mean', df_name='pos_cash_bal')


# ### 3. Prepare applications train/test for merge and training:

# In[ ]:


def categorical_features_factorize(X):

    categorical_feats = [col for col in X.columns if X[col].dtype == 'object']
    print('Categorical features encoding: {}'.format(categorical_feats))

    for col in categorical_feats:
        X[col] = pd.factorize(X[col])[0]

    print('Categorical features encoded.')

    return X


# In[ ]:


# Concatenate train and test.
X = pd.concat([train, test], ignore_index=True, sort=False)

# Encode categorical features in concatenated DF.
X = categorical_features_factorize(X)


# ### 4. Merge all prepared DFs onto concatenated train/test DF:

# In[ ]:


X = X.merge(bureau_prepared, how='left', on='SK_ID_CURR', copy=False)
X = X.merge(prev_prepared, how='left', on='SK_ID_CURR', copy=False)
X = X.merge(ins_prepared, how='left', on='SK_ID_CURR', copy=False)
X = X.merge(pos_cash_bal_prepared, how='left', on='SK_ID_CURR', copy=False)
X = X.merge(cred_card_bal_prepared, how='left', on='SK_ID_CURR', copy=False)


# ### 5. Split into train and test DFs once again and remove unnecessary columns:

# In[ ]:


# Split data into train and test once again, based on availability of TARGET variable.
X_train = X[X['TARGET'].notnull()]
X_test = X[X['TARGET'].isnull()]

# Select TARGET and create a new variable for it, useful for model training.
y_train = X_train.TARGET

# Remove X (concatenated DF), as it will not be needed anymore.
del X
gc.collect()

# Select only features relevant to the model, do not use ID or index ones!
good_features = [x for x in X_train.columns if x not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]


# ### 6. KFold split and model training:
# 
# Preparing a function for training in this way is useful for a number of reasons:
# - it is easy to change model parameters
# - it is easy to change number of folds
# - it is easy to change split seed to train on different splits
# 
# Therefore a number of runs with very different parameters can be easily created for proper ensembling.

# In[ ]:


def run_kfold_lgbm(X_train,
                   y_train,
                   X_test,
                   model_params,
                   n_folds=5,
                   seed=1337):
    
    
    # Prepare KFold split, Stratified works well in this competition.
    # Parametrize it's seed to enable easy change of splits.
    kf = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=seed)
    
    # Subset features to eliminate irrelevant ones.
    X_train = X_train[good_features]
    X_test = X_test[good_features]
    
    # Assert that both train and test have the same set of columns.
    assert np.all(X_train.columns == X_test.columns), '    Train and test sets must have the same set of columns.'

    # Create oof sets for prediction storage.
    # Create gbm_history for storage of best AUC per fold.
    oof_train = np.zeros((X_train.shape[0]))
    oof_test = np.zeros((X_test.shape[0], n_folds))
    gbm_history = {}

    # Helper variable to index oof
    i = 0
    
    for train_index, valid_index in kf.split(X=X_train, y=y_train):
        assert len(np.intersect1d(train_index, valid_index)) == 0, '        Train and test indices must not overlap.'
        
        print('Running on fold: {}'.format(i + 1))

        # Create train and validation sets based on KFold indices.
        X_tr = X_train.iloc[train_index]
        X_val = X_train.iloc[valid_index]
        y_tr = y_train.iloc[train_index]
        y_val = y_train.iloc[valid_index]

        dtrain = lgb.Dataset(X_tr, y_tr)
        dvalid = lgb.Dataset(X_val, y_val, reference=dtrain)

        # Train LightGBM model, it's parameters can be changed easily
        # through model_params function variable.
        gbm = lgb.train(
            params=model_params,
            train_set=dtrain,
            evals_result=gbm_history,
            num_boost_round=10000,
            valid_sets=[dtrain, dvalid],
            early_stopping_rounds=200,
            verbose_eval=100)

        # Predict validation and test data and store them in oof sets.
        oof_train[valid_index] = gbm.predict(
            X_val, num_iteration=gbm.best_iteration)
        oof_test[:, i] = gbm.predict(X_test, num_iteration=gbm.best_iteration)
        
        # Show best AUC per fold based on GBM training history.
        print('Best fold GBM AUC: {:.4f}\n'.format(np.max(gbm_history['valid_1']['auc'])))

        i += 1

    return oof_train, oof_test


# In[ ]:


gbm_params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'nthread': 6,
    'learning_rate': 0.05,  # 02,
    'num_leaves': 20,
    'colsample_bytree': 0.9497036,
    'subsample': 0.8715623,
    'subsample_freq': 1,
    'max_depth': 8,
    'reg_alpha': 0.041545473,
    'reg_lambda': 0.0735294,
    'min_split_gain': 0.0222415,
    'min_child_weight': 60, # 39.3259775,
    'seed': 0,
    'verbose': -1,
    'metric': 'auc',
}


oof_train, oof_test = run_kfold_lgbm(X_train, y_train, X_test, gbm_params)


# In[ ]:


# Take mean of fold predictions for the test data.
submission_preds = oof_test.mean(axis=1)

# Prepare submission format and save it.
submission_df = X_test[['SK_ID_CURR']].copy()
submission_df['TARGET'] = submission_preds
# submission_df[['SK_ID_CURR', 'TARGET']].to_csv('pipeline_lgbm.csv', index= False)

