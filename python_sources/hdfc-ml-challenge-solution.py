#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df_train = pd.read_csv('/kaggle/input/hdfc-2019/DataSet/Train.csv')
df_test = pd.read_csv('/kaggle/input/hdfc-2019/DataSet/Test.csv')


# In[ ]:


df_train.shape, df_test.shape


# In[ ]:


df_train.head()


# # Col1 has an unique data & Col2 is Target variable

# In[ ]:


df_test.head()


# In[ ]:


X = df_train.drop(['Col1','Col2'], axis=1)
y = df_train.Col2

XTest = df_test.drop(['Col1'], axis=1)


# In[ ]:


y.value_counts(normalize=True) * 100


# In[ ]:





# # More information about train and test dataframe

# In[ ]:


X.info()


# In[ ]:


XTest.info()


# # Something is fishy:
# > **X** has 2 object type features but **XTest** has 11 object type features
# 
# > Ideally X and XTest dataframe should contain features with same datatype.

# In[ ]:


for col in X.select_dtypes('object').columns:
    print(col)
    print(X[col].unique())
    print("--------------------------------------------------------------------\n")


# In[ ]:


for col in XTest.select_dtypes('object').columns:
    print(col)
    print(XTest[col].unique())
    print("--------------------------------------------------------------------\n")


# # Clean the data:
# 
# > * We will convert string numerical data to float.
# 
# > * We will convert '-' sign to nan
# 
# 

# In[ ]:


def convert_to_float(row):
    if row == '-':
        return np.nan
    else:
        return float(row)


# In[ ]:


columns_need_treatment = list(X.select_dtypes('object').columns) + list(XTest.select_dtypes('object').columns)
print(len(columns_need_treatment))
print(columns_need_treatment)


# In[ ]:


for col in columns_need_treatment:
    X[col] = X[col].apply(convert_to_float)
    XTest[col] = XTest[col].apply(convert_to_float)


# In[ ]:





# # Does data contains duplicate rows or features ?
# 

# In[ ]:


duplicate_rows_in_train = X.duplicated()
duplicate_rows_in_test = XTest.duplicated()

print("Train data contains %d duplicate rows and Test data contains %d duplicate rows."%(sum(duplicate_rows_in_train), 
                                                                                             sum(duplicate_rows_in_test)))


# In[ ]:


y[duplicate_rows_in_train].value_counts(normalize=True)


# In[ ]:


y[duplicate_rows_in_train].head(30)


# In[ ]:


X['duplicate_row'] = False
XTest['duplicate_row'] = False

X.loc[duplicate_rows_in_train, 'duplicate_row'] = True
XTest.loc[duplicate_rows_in_test, 'duplicate_row'] = True


# In[ ]:


features = X.columns
duplicate_columns = set()
for i in range(len(features)):
    for j in range(i+1, len(features)):
        if np.all(X[features[i]] == X[features[j]]):
            print(features[i], features[j])
            duplicate_columns.add(features[j])


# In[ ]:


print("Number of duplicate columns:",len(duplicate_columns))


# In[ ]:


selected_features = [_ for _ in X.columns if _ not in duplicate_columns]
print("Selected_Features :",len(selected_features))


# In[ ]:


RANDOM_SEED = 1


# In[ ]:


from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score
import xgboost as xgb
import lightgbm as lgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.model_selection import StratifiedKFold


# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X[selected_features], y, 
                                                      random_state=RANDOM_SEED, 
                                                      test_size=0.2)


# In[ ]:


def score(params):
    try:

        print("Training with params: ",params)
        num_round = int(params['n_estimators'])
        del params['n_estimators']
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        watchlist = [(dtrain, 'train'),(dvalid, 'eval')]
        gbm_model = xgb.train(params, dtrain, num_round,
                              evals=watchlist,
                              verbose_eval=False)
        predictions = gbm_model.predict(dvalid,
                                        ntree_limit=gbm_model.best_iteration + 1)
        predictions = (predictions >= 0.5).astype('int')
        score = f1_score(y_valid, predictions, average='weighted')
        print("\tScore {0}\n\n".format(score))
        
        # The score function should return the loss (1-score)
        # since the optimize function looks for the minimum
        loss = 1 - score
        return {'loss': loss, 'status': STATUS_OK}
   
    # In case of any exception or assertionerror making score 0, so that It can return maximum loss (ie 1)
    except AssertionError as obj:
        #print("AssertionError: ",obj)
        loss = 1 - 0
        return {'loss': loss, 'status': STATUS_OK}

    except Exception as obj:
        #print("Exception: ",obj)
        loss = 1 - 0
        return {'loss': loss, 'status': STATUS_OK}


# In[ ]:


def optimize(
             trials, 
             max_evals, 
             random_state=RANDOM_SEED):


    """
    This is the optimization function that given a space (space here) of 
    hyperparameters and a scoring function (score here), finds the best hyperparameters.
    """
    # To learn more about XGBoost parameters, head to this page: 
    # https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
    space = {
        'n_estimators': hp.quniform('n_estimators', 100, 300, 1),
        'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
        # A problem with max_depth casted to float instead of int with
        # the hp.quniform method.
        'max_depth':  hp.choice('max_depth', np.arange(1, 7, dtype=int)),
        'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
        'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
        'gamma': hp.quniform('gamma', 0, 1, 0.05),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
        'scale_pos_weight': hp.quniform('scale_pos_weight', 1,4, 0.05),
        "reg_alpha": hp.quniform('reg_alpha', 0, 1, 0.05),
        "reg_lambda": hp.quniform('reg_lambda', 1, 5, 0.05),
        'eval_metric': 'logloss',
        'objective': 'binary:logistic',
        # Increase this number if you have more cores. Otherwise, remove it and it will default 
        # to the maxium number. 
        'nthread': 4,
        'booster': 'gbtree',
        'tree_method': 'exact',
        'silent': 1,
        'seed': random_state
    }
    # Use the fmin function from Hyperopt to find the best hyperparameters
    best = fmin(score, 
                space, 
                algo=tpe.suggest, 
                trials=trials, 
                max_evals=max_evals)
    return best


# In[ ]:


trials = Trials()
MAX_EVALS = 25

best_hyperparams = optimize(trials, MAX_EVALS)
print("The best hyperparameters are: ", "\n")
print(best_hyperparams)


# # Best Hyper-parameters

# In[ ]:


best_hyperparams


# In[ ]:


param = best_hyperparams
num_round = int(param['n_estimators'])
del param['n_estimators']


# In[ ]:





# # OOF (Out of Fold Prediciton): 
# 

# In[ ]:


num_splits = 5
skf = StratifiedKFold(n_splits= num_splits, random_state= RANDOM_SEED, shuffle=True)


# In[ ]:


dxtest = xgb.DMatrix(XTest[selected_features])


# In[ ]:


y_test_pred = np.zeros((XTest[selected_features].shape[0], 1))
print(y_test_pred.shape)
y_valid_scores = []

X_TRAIN = X[selected_features].copy()
Y_TRAIN = y.copy()
X_TRAIN = X_TRAIN.reindex()
Y_TRAIN = Y_TRAIN.reindex()

fold_cnt = 1
for train_index, test_index in skf.split(X_TRAIN,Y_TRAIN):
    print("FOLD .... ",fold_cnt)
    fold_cnt += 1
    
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_valid = X_TRAIN.iloc[train_index], X_TRAIN.iloc[test_index]
    y_train, y_valid = Y_TRAIN.iloc[train_index], Y_TRAIN.iloc[test_index]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    
    evallist = [(dtrain, 'train'), (dvalid, 'eval')]

    # Training xgb model
    bst = xgb.train(param, dtrain, num_round, evallist, verbose_eval=50)
    
    # Predict Validation
    y_pred_valid = bst.predict(dvalid, ntree_limit=bst.best_iteration + 1)
    y_valid_scores.append(f1_score(y_valid, (y_pred_valid >= 0.5).astype(int), average='weighted'))
   
    # Predict Test 
    y_pred = bst.predict(dxtest, ntree_limit=bst.best_iteration+1)
    
    y_test_pred += y_pred.reshape(-1,1)

#Normalize test predicted probability
y_test_pred /= num_splits


# In[ ]:


y_valid_scores


# In[ ]:


print("Average validation_score: ",np.mean(y_valid_scores))


# In[ ]:


output = df_test[['Col1']].copy()
output['Col2'] = (y_test_pred >= 0.5).astype(int)


# In[ ]:


output.head()


# In[ ]:


output['Col2'].value_counts()/output.shape[0] * 100


# In[ ]:


output.to_csv("./predict_hdfc_xgb_oof.csv", index=False)


# In[ ]:




