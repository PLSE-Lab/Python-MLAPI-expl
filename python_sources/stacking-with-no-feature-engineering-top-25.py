#!/usr/bin/env python
# coding: utf-8

# This notebook is focused on a final "stacking" part of DS work on a project. 
# 
# The goal is to show the value of model stacking to the overall score.
# At the moment of committing, this kernel gets top 25% result with no feature engineering at all. 
# 
# Although, as FE is the most important part of DS work, I hope that this kernel as a template for your modelling part will help you to focus on feature engineering.

# ## Installs and reading data

# In[ ]:


# importing libraries
import pandas as pd
import numpy as np

import warnings
warnings.simplefilter('ignore')

# setting seed
seed = 11


# In[ ]:


# reading files
train = pd.read_csv('../input/learn-together/train.csv', index_col="Id")
test = pd.read_csv('../input/learn-together/test.csv')

# deleting columns with no nulls (found previously)
del train['Soil_Type15']
del train['Soil_Type7']

del test['Soil_Type15']
del test['Soil_Type7']


# In[ ]:


# defining features and target
features = train.drop(['Cover_Type'], axis=1)
target = train.Cover_Type


# ## Defining models
# Firstly, we choose models to stack.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

RF_model = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
                       max_depth=24, max_features=17, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=138,
                       n_jobs=None, oob_score=False, random_state=11, verbose=0,
                       warm_start=False)


# In[ ]:


from lightgbm import LGBMClassifier

LGBM_model = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
               importance_type='split', learning_rate=0.1, max_depth=-1,
               min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
               n_estimators=198, n_jobs=-1, num_leaves=64, objective=None,
               random_state=11, reg_alpha=0.0, reg_lambda=0.0, silent=True,
               subsample=1.0, subsample_for_bin=200000, subsample_freq=0)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

KN_model = KNeighborsClassifier(algorithm='kd_tree', leaf_size=61, metric='minkowski',
                     metric_params=None, n_jobs=-1, n_neighbors=2, p=1,
                     weights='distance')


# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier

ExtraTr_model = ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
                     max_depth=None, max_features=21, max_leaf_nodes=None,
                     min_impurity_decrease=0.0, min_impurity_split=None,
                     min_samples_leaf=1, min_samples_split=4,
                     min_weight_fraction_leaf=0.0, n_estimators=420, n_jobs=-1,
                     oob_score=False, random_state=11, verbose=0,
                     warm_start=False)


# In[ ]:


models_list = [
    RF_model, 
    LGBM_model, 
    KN_model, 
    ExtraTr_model
]


# ## Stacking

# As a stacking function I use StackingCVClassifier. For me it was most simple and convinient

# In[ ]:


from mlxtend.classifier import StackingCVClassifier

sclf = StackingCVClassifier(classifiers=models_list,
                            meta_classifier=RandomForestClassifier(n_estimators=500, random_state=seed),
                            cv=5,
                            random_state=seed,
                            verbose=1, 
                            n_jobs=-1)


# In[ ]:


# fitting metamodel
sclf.fit(features, target)


# In[ ]:


# applying the metamodel to the test data and getting predictions
test_pred = sclf.predict(test.drop('Id', axis=1))


# ## Exporting to file

# In[ ]:


# making a dataframe with a result set
output = pd.DataFrame({'ID': test.Id,
                       'Cover_Type': test_pred})

# exporting result dataframe to csv
output.to_csv('stacking submission.csv', index=False)
print('Export done')

