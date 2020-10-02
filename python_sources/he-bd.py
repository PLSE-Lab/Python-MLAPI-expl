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
# importing dataset
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 15:11:34 2018

@author: Ankur
"""

# importing dataset
import pandas as pd

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_owner = pd.read_csv('../input/Building_Ownership_Use.csv')
df_struct = pd.read_csv('../input/Building_Structure.csv')
merge = pd.merge(df_struct, df_owner, on='building_id')
train = pd.merge(df_train, merge, on='building_id')
test = pd.merge(df_test, merge, on='building_id')

# deleting dataset to free up the space
del df_train, df_test, df_owner, df_struct, merge

# mapping categorical feature data
import numpy as np
area_assesed_mapping = {label:idx for idx,label in enumerate(np.unique(train['area_assesed']))}
train['area_assesed'] = train['area_assesed'].map(area_assesed_mapping)
test['area_assesed'] = test['area_assesed'].map(area_assesed_mapping)

land_surface_condition_mapping = {label:idx for idx,label in enumerate(np.unique(train['land_surface_condition']))}
train['land_surface_condition'] = train['land_surface_condition'].map(land_surface_condition_mapping)
test['land_surface_condition'] = test['land_surface_condition'].map(land_surface_condition_mapping)

foundation_type_mapping = {label:idx for idx,label in enumerate(np.unique(train['foundation_type']))}
train['foundation_type'] = train['foundation_type'].map(foundation_type_mapping)
test['foundation_type'] = test['foundation_type'].map(foundation_type_mapping)

roof_type_mapping = {label:idx for idx,label in enumerate(np.unique(train['roof_type']))}
train['roof_type'] = train['roof_type'].map(roof_type_mapping)
test['roof_type'] = test['roof_type'].map(roof_type_mapping)

ground_floor_type_mapping = {label:idx for idx,label in enumerate(np.unique(train['ground_floor_type']))}
train['ground_floor_type'] = train['ground_floor_type'].map(ground_floor_type_mapping)
test['ground_floor_type'] = test['ground_floor_type'].map(ground_floor_type_mapping)

other_floor_type_mapping = {label:idx for idx,label in enumerate(np.unique(train['other_floor_type']))}
train['other_floor_type'] = train['other_floor_type'].map(other_floor_type_mapping)
test['other_floor_type'] = test['other_floor_type'].map(other_floor_type_mapping)

position_mapping = {label:idx for idx,label in enumerate(np.unique(train['position']))}
train['position'] = train['position'].map(position_mapping)
test['position'] = test['position'].map(position_mapping)

plan_configuration_mapping = {label:idx for idx,label in enumerate(np.unique(train['plan_configuration']))}
train['plan_configuration'] = train['plan_configuration'].map(plan_configuration_mapping)
test['plan_configuration'] = test['plan_configuration'].map(plan_configuration_mapping)

condition_post_eq_mapping = {label:idx for idx,label in enumerate(np.unique(train['condition_post_eq']))}
train['condition_post_eq'] = train['condition_post_eq'].map(condition_post_eq_mapping)
test['condition_post_eq'] = test['condition_post_eq'].map(condition_post_eq_mapping)

legal_ownership_status_mapping = {label:idx for idx,label in enumerate(np.unique(train['legal_ownership_status']))}
train['legal_ownership_status'] = train['legal_ownership_status'].map(legal_ownership_status_mapping)
test['legal_ownership_status'] = test['legal_ownership_status'].map(legal_ownership_status_mapping)

class_mapping = {label:idx for idx,label in enumerate(np.unique(train['damage_grade']))}
train['damage_grade'] = train['damage_grade'].map(class_mapping)

# setting building_id as index
train.set_index('building_id', inplace=True)
test.set_index('building_id', inplace=True)

# Imputing missing values
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr1 = imr.fit(train)
train_imputed = imr1.transform(train.values)
imr2 = imr.fit(test)
test_imputed = imr2.transform(test.values)

del train, test

# seperating features and class labels
x1, x2 = train_imputed[:,0], train_imputed[:,2:]
x1 = x1[:, np.newaxis]
X_train = np.concatenate((x1, x2), axis=1)
y = train_imputed[:,1]
X_test1 = test_imputed[:,:]

del train_imputed, test_imputed, x1, x2

# partitioning the train dataset
from sklearn.model_selection import train_test_split
X_training, X_testing, y_training, y_testing = train_test_split(X_train, y, test_size=0.3, random_state=0)

# Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import Pipeline

#pipe_rf = Pipeline([('Scaler', StandardScaler()),
#                    ('Feature_Selection', SelectFromModel(ExtraTreesClassifier)),
#                    ('Classifier', RandomForestClassifier())])
clf = RandomForestClassifier(n_estimators=20)

# Utility function to report best scores
import numpy as np
from time import time
from scipy.stats import randint as sp_randint

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        canditates = np.flatnonzero(results['rank_test_score'] == i)
        for canditate in canditates:
            print("Model with rank: {0}".format(i))
            print("mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results['mean_test_score'][canditate],
                    results['std_test_score'][canditate]))
            print("Parameters: {0}".format(results['params'][canditate]))
            print("")
            
            
# specify parameters and distributions to sample form
param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 14),
              "min_samples_split": sp_randint(2, 14),
              "min_samples_leaf": sp_randint(1, 14),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run randomized search
from sklearn.model_selection import RandomizedSearchCV
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)

start = time()
random_search.fit(X_training, y_training)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)

# use a full grid over all parameters
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 13],
              "min_samples_split": [2, 3, 13],
              "min_samples_leaf": [1, 3, 13],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run grid search
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time()
grid_search.fit(X_training, y_training)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)

# Best training scores and parameters of both CV search
print("\nRandom Search CV training score:\n")
print(random_search.best_score_)
print(random_search.best_params_)
print('\nGrid Search CV training score:\n')
print(grid_search.best_score_)
print(grid_search.best_params_)

# Testing scores
clf = random_search.best_estimator_
clf.fit(X_training, y_training)
print('\nRandomSearchCV Test Accuracy: %.3f' % clf.score(X_testing, y_testing))

from sklearn.metrics import confusion_matrix
y_pred_rs = clf.predict(X_testing)
confmat_rs = confusion_matrix(y_true=y_testing, y_pred=y_pred_rs)
print(confmat_rs)
from sklearn.metrics import precision_score, recall_score, f1_score
print('Precision: %.3f' % precision_score(y_true=y_testing, y_pred=y_pred_rs, average="weighted"))
print('Recall: %.3f' % recall_score(y_true=y_testing, y_pred=y_pred_rs, average="weighted"))
print('F1: %.3f' % f1_score(y_true=y_testing, y_pred=y_pred_rs, average="weighted"))


clf = grid_search.best_estimator_
clf.fit(X_training, y_training)
print('GridSearchCV Test Accuracy: %.3f' % clf.score(X_testing, y_testing))

y_pred_gs = clf.predict(X_testing)
confmat_gs = confusion_matrix(y_true=y_testing, y_pred=y_pred_gs)
print(confmat_gs)
print('Precision: %.3f' % precision_score(y_true=y_testing, y_pred=y_pred_gs, average="weighted"))
print('Recall: %.3f' % recall_score(y_true=y_testing, y_pred=y_pred_gs, average="weighted"))
print('F1: %.3f' % f1_score(y_true=y_testing, y_pred=y_pred_gs, average="weighted"))