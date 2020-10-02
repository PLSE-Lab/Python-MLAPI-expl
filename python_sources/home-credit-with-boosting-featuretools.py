#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import pandas_profiling
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import FunctionTransformer, StandardScaler
import sklearn
import featuretools as ft

import os
print(os.listdir("../input"))


# In[ ]:


get_ipython().run_line_magic('env', 'JOBLIB_TEMP_FOLDER=/tmp')


# In[ ]:


application_train = pd.read_csv('../input/application_train.csv')
application_train.head(10)


# In[ ]:


application_train_X = application_train.drop('TARGET',1)
train_Y = application_train['TARGET']


# In[ ]:


clf = DummyClassifier()
param_grid = {'strategy': ['stratified']}
model = GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1, cv=7, scoring='roc_auc')
model.fit(application_train_X[['SK_ID_CURR']], train_Y)

print('DummyClassifier...')
print('Best Params:')
print(model.best_params_)
print('Best CV Score:')
print(model.best_score_)


# In[ ]:


def get_cat_cols(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'] 
    newdf = df.select_dtypes(exclude=numerics)
    return newdf.columns


# In[ ]:


def get_number_cat_col(df):
    all_col = list(df.columns)
    num_cat_col = []
    cat_col = list(get_cat_cols(df))
    for i in range(len(all_col)):
        if all_col[i] in cat_col:
            num_cat_col.append(i)
            
    return num_cat_col


# In[ ]:


cbc = CatBoostClassifier(eval_metric='AUC',cat_features=get_number_cat_col(application_train_X))


# In[ ]:


param_grid = {}
model = GridSearchCV(estimator=cbc, param_grid=param_grid, n_jobs=-1, cv=2, scoring='roc_auc')
model.fit(application_train_X.fillna(0), train_Y)

print('CatBoostClassifier...')
print('Best Params:')
print(model.best_params_)
print('Best CV Score:')
print(model.best_score_)


# In[ ]:


class GetDummies(sklearn.base.TransformerMixin):
    """Fast one-hot-encoder that makes use of pandas.get_dummies() safely
    on train/test splits.
    """
    def __init__(self, dtypes=None):
        self.input_columns = None
        self.final_columns = None
        if dtypes is None:
            dtypes = [object, 'category']
        self.dtypes = dtypes

    def fit(self, X, y=None, **kwargs):
        self.input_columns = list(X.select_dtypes(self.dtypes).columns)
        X = pd.get_dummies(X, columns=self.input_columns)
        self.final_columns = X.columns
        return self
        
    def transform(self, X, y=None, **kwargs):
        X = pd.get_dummies(X, columns=self.input_columns)
        X_columns = X.columns
        # if columns in X had values not in the data set used during
        # fit add them and set to 0
        missing = set(self.final_columns) - set(X_columns)
        for c in missing:
            X[c] = 0
        # remove any new columns that may have resulted from values in
        # X that were not in the data set when fit
        return X[self.final_columns]
    
    def get_feature_names(self):
        return tuple(self.final_columns)


# In[ ]:


#%%time
get_dummies = GetDummies()
#application_train_X = get_dummies.fit_transform(application_train_X)


# In[ ]:


get_ipython().run_line_magic('time', '')
cbc = CatBoostClassifier(eval_metric='AUC')
param_grid = {}
model = GridSearchCV(estimator=cbc, param_grid=param_grid, n_jobs=-1, cv=2, scoring='roc_auc')
model.fit(application_train_X.fillna(0), train_Y)

print('CatBoostClassifier...')
print('Best Params:')
print(model.best_params_)
print('Best CV Score:')
print(model.best_score_)


# In[ ]:


def do_scalar(df):
    scaler = StandardScaler()
    scaler.fit(df)
    df = scaler.transform(df)
    return df


# In[ ]:


application_train_X = get_dummies.fit_transform(application_train_X)
application_train_X_scalar = do_scalar(application_train_X.fillna(0))


# In[ ]:


get_ipython().run_cell_magic('time', '', "rc = RidgeClassifier()\nparam_grid = {}\nmodel = GridSearchCV(estimator=rc, param_grid=param_grid, n_jobs=-1, cv=2, scoring='roc_auc')\nbest_model = model.fit(application_train_X_scalar, train_Y)\n\nprint('RidgeClassifier...')\nprint('Best Params:')\nprint(model.best_params_)\nprint('Best CV Score:')\nprint(model.best_score_)")


# In[ ]:


def transform_columns(df,columns):
    for col in columns:
        if not col in df.columns:
            df[col]=0
    for col in df.columns:
        if not col in columns:
             df = df.drop(col,1)
                
    return df


# In[ ]:


application_test_X = pd.read_csv('../input/application_test.csv')
application_test_X = get_dummies.fit_transform(application_test_X)
application_test_X = transform_columns(application_test_X,application_train_X.columns)
application_test_X_scalar = do_scalar(application_test_X.fillna(0))
SK_ID_CURR_Test = application_test_X['SK_ID_CURR']
TARGET_Test = best_model.predict(application_test_X_scalar)
submission_df = {"SK_ID_CURR ": SK_ID_CURR_Test,
                 "TARGET ": TARGET_Test}
submission = pd.DataFrame(submission_df)
submission.to_csv('submission_ridge.csv', index=False)


# In[ ]:


bureau = pd.read_csv('../input/bureau.csv')


# In[ ]:


bureau.info()


# In[ ]:


application_train = pd.read_csv('../input/application_train.csv')


# In[ ]:


es = ft.EntitySet(id = 'clients')


# In[ ]:


es = es.entity_from_dataframe(entity_id = 'app', dataframe = application_train, index = 'SK_ID_CURR')


# In[ ]:


es = es.entity_from_dataframe(entity_id = 'bureau', dataframe = bureau, index = 'SK_ID_BUREAU')


# In[ ]:


r_app_bureau = ft.Relationship(es['app']['SK_ID_CURR'], es['bureau']['SK_ID_CURR'])


# In[ ]:


#Add in the defined relationships
es = es.add_relationships([r_app_bureau])
# Print out the EntitySet
es


# In[ ]:


# Default primitives from featuretools
default_agg_primitives =  ["sum", "std", "max", "skew", "min", "mean", "count", "percent_true", "num_unique", "mode"]
default_trans_primitives =  ["day", "year", "month", "weekday", "haversine", "numwords", "characters"]


# In[ ]:


# DFS with specified primitives
feature_names = ft.dfs(entityset = es, target_entity = 'app',
                       trans_primitives = default_trans_primitives,
                       agg_primitives=default_agg_primitives, 
                       max_depth = 2, features_only=True)

print('%d Total Features' % len(feature_names))


# In[ ]:


feature_matrix, feature_names = ft.dfs(entityset = es, target_entity = 'app',
                                       trans_primitives = default_trans_primitives,
                                       agg_primitives=default_agg_primitives, 
                                        max_depth = 2, features_only=False, verbose = True,n_jobs = -1)


# In[ ]:


application_train_X = feature_matrix.drop('TARGET',1)
train_Y = feature_matrix['TARGET']


# In[ ]:


categorical_features_indices = np.where(application_train_X.dtypes != np.float)[0]


# In[ ]:


cbc = CatBoostClassifier(eval_metric='AUC',cat_features=get_number_cat_col(application_train_X))
param_grid = {}
model = GridSearchCV(estimator=cbc, param_grid=param_grid, n_jobs=-1, cv=2, scoring='roc_auc')
model.fit(application_train_X.fillna(0), train_Y)

print('CatBoostClassifier...')
print('Best Params:')
print(model.best_params_)
print('Best CV Score:')
print(model.best_score_)


# In[ ]:


previous_app  = pd.read_csv('../input/previous_application.csv')


# In[ ]:


es = es.entity_from_dataframe(entity_id = 'previous_app', dataframe = previous_app, index = 'SK_ID_PREV')


# In[ ]:


r_app_previous_app = ft.Relationship(es['app']['SK_ID_CURR'], es['previous_app']['SK_ID_CURR'])


# In[ ]:


#Add in the defined relationships
es = es.add_relationships([r_app_previous_app])
# Print out the EntitySet
es


# In[ ]:


# DFS with specified primitives
feature_names = ft.dfs(entityset = es, target_entity = 'app',
                       trans_primitives = default_trans_primitives,
                       agg_primitives=default_agg_primitives, 
                       max_depth = 2, features_only=True)

print('%d Total Features' % len(feature_names))


# In[ ]:


feature_matrix, feature_names = ft.dfs(entityset = es, target_entity = 'app',
                                       trans_primitives = default_trans_primitives,
                                       agg_primitives=default_agg_primitives, 
                                        max_depth = 2, features_only=False, verbose = True,n_jobs = -1)


# In[ ]:


application_train_X = feature_matrix.drop('TARGET',1)
train_Y = feature_matrix['TARGET']


# In[ ]:


cbc = CatBoostClassifier(eval_metric='AUC',cat_features=get_number_cat_col(application_train_X))
param_grid = {}
model = GridSearchCV(estimator=cbc, param_grid=param_grid, n_jobs=-1, cv=2, scoring='roc_auc')
model.fit(application_train_X.fillna(0), train_Y)

print('CatBoostClassifier...')
print('Best Params:')
print(model.best_params_)
print('Best CV Score:')
print(model.best_score_)

