#!/usr/bin/env python
# coding: utf-8

# Titanic
# =======
# This is a learning and experimentation kernel in order to implement and understand the basics of ensembling and stacking.  
# Most of the work in this notebook is guided by Anisotropic's [Introduction to Ensembling/Stacking in Python](https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python) notebook. I have, however, made a small attempt to take the ideas presented and generalize them code-wise.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import os
import re
from bisect import bisect

import xgboost as xgb

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn import metrics


# In[ ]:


data_dir = "../input"
os.listdir(f"{data_dir}")


# Data Exploratoration
# --------------------

# In[ ]:


# loading the raw datasets
train_df_raw = pd.read_csv(f'{data_dir}/train.csv', low_memory=False)
test_df_raw = pd.read_csv(f'{data_dir}/test.csv', low_memory=False)

# just to peek at the raw data
train_df_raw.sample(10)


# In[ ]:


train_df_raw.describe(include='all').T


# In[ ]:


test_df_raw.describe(include='all').T


# Initial Cleanup
# ---------------
# Based on the simple exploration it is immediately obvious that data is missing from both the training and test sets. Additonally, there are categorical columns that will need to be converted.

# In[ ]:


train_df = train_df_raw
test_df = test_df_raw

all_df = [train_df, test_df]


# ### Fill Missing Variables
# Filling in the missing data with the best guess I could come up with.
# - **Embarked**: 'S' was the top result
# - **Age, Fare**: just using the statistical mean
# - **Cabin**: filling with empty, can this be figured out by ticket?

# In[ ]:


for df in all_df:
    df['Embarked'] = df['Embarked'].fillna('S')
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Cabin'] = df['Cabin'].fillna('')
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())


# ### Categorize Variables
# naively converting categories to whatever their index would be via pd.DataFrame.uniques()  
# I kept `category_maps` around as a dictionary of dictionarys of the conversions that take place, but it's only there for reference and debugging. 

# In[ ]:


# accumulated list of maps of categories that are created
category_maps = {}

# todo: to generalize well we should choose better storage types
def Categorify(df:pd.DataFrame, cat_names):
    for cat_name in cat_names:
        uniques = df[cat_name].unique()
        category_maps[cat_name] = {i:uniques[i] for i in range(len(uniques))}
        df[cat_name] = [np.where(uniques == key)[0][0] for key in df[cat_name]]


# In[ ]:


cat_names = ['Sex', 'Embarked']
list(map(lambda df: Categorify(df, cat_names), all_df))

category_maps


# ### Quantiles

# In[ ]:


def Quantile(df:pd.DataFrame, quant_names, quants = [0.25,0.5,0.75]):
    for quant_name in quant_names:
        quant_col_name = f'{quant_name}_quantile'
        quant_vals = [np.quantile(df[quant_name], quant) for quant in quants]
        df[quant_col_name] = [bisect(quant_vals, x) for x in df[quant_name]]


# In[ ]:


get_ipython().run_cell_magic('capture', '', "quant_names = ['Fare', 'Age']\nlist(map(lambda df: Quantile(df, quant_names), all_df))")


# Create New Features
# -------------------
# keying off of other notebooks as far as new feature generation goes:
#  - **Title**:
#  - **FamilySize**:
#  - **IsAlone**:
#  - **HasCabin**:
#  - **Fare_quantile**:
#  - **Age_quantile**:

# In[ ]:


# extracts a title
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

# title mappings, if it's not in this list then it will be marked as 'Rare'
title_map = {
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Mme": "Mrs",
    "Miss": "Miss",
    "Mlle": "Miss",
    "Ms": "Miss",
    "Master": "Master"
}


# In[ ]:


# looping through training and test sets to apply new feature generation
for df in all_df:
    # Title
    df['Title'] = list(map(lambda key: title_map.get(key, 'Rare'), 
                                       df['Name'].apply(get_title)))
    Categorify(df, ['Title'])
    
    # Family Size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Is Alone
    df['IsAlone'] = [1 if size == 1 else 0 for size in df['FamilySize']]
    
    # Has Cabin
    df['HasCabin'] = [0 if cabin == '' else 1 for cabin in df['Cabin']]


# Base Model Stacking
# -------------------

# ### Classifier Helper

# In[ ]:


SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(n_splits=NFOLDS, random_state=SEED)

# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, name, clf, seed=0, params=None):
        params['random_state'] = seed
        self.name = name
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        return self.clf.fit(x,y).feature_importances_


# In[ ]:


def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((x_train.shape[0],))
    oof_test = np.zeros((x_test.shape[0],))
    oof_test_skf = np.empty((NFOLDS, x_test.shape[0]))

    for i, (train_index, test_index) in enumerate(kf.split(x_train, y_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# ### Hyperparameters

# In[ ]:


# create the classifiers
classifier_stack = [
    SklearnHelper('RandomForest', clf=RandomForestClassifier, seed=SEED, params={
        'n_jobs': -1,
        'n_estimators': 500,
         #'max_features': 0.2,
        'max_depth': 6,
        'min_samples_leaf': 2,
        'max_features' : 'sqrt',
        'verbose': 0
    }),
    SklearnHelper('ExtraTrees', clf=ExtraTreesClassifier, seed=SEED, params={
        'n_jobs': -1,
        'n_estimators':500,
        #'max_features': 0.5,
        'max_depth': 8,
        'min_samples_leaf': 2,
        'verbose': 0
    }),
    SklearnHelper('AdaBoost', clf=AdaBoostClassifier, seed=SEED, params={
        'n_estimators': 500,
        'learning_rate' : 0.75
    }),
    SklearnHelper('GradientBoost', clf=GradientBoostingClassifier, seed=SEED, params={
        'n_estimators': 500,
         #'max_features': 0.2,
        'max_depth': 5,
        'min_samples_leaf': 2,
        'verbose': 0
    })
]


# In[ ]:


dep_var = 'Survived'
drop_vars = ['PassengerId', 'Name', 'Ticket', 'Cabin']

x_train_df = train_df.drop(drop_vars, axis=1).drop(dep_var, axis=1)
x_train = x_train_df.values
y_train = train_df[dep_var].ravel()
x_test = test_df.drop(drop_vars, axis=1).values


# In[ ]:


# train the classifier via the out-of-fold 
oofs = { clf.name : get_oof(clf, x_train, y_train, x_test) for clf in classifier_stack }
print("Training is complete")


# In[ ]:


features = [map(lambda clf: clf.feature_importances(x_train,y_train), classifier_stack)]


# Second Level Stack
# ------------------

# In[ ]:


base_predictions_train = pd.DataFrame({ key: oofs[key][0].ravel() for key in oofs })
base_predictions_train.head()


# In[ ]:


x_train = np.concatenate([oofs[key][0] for key in oofs], axis=1)
x_test = np.concatenate([oofs[key][1] for key in oofs], axis=1)


# In[ ]:


gbm = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1).fit(x_train, y_train)
predictions = gbm.predict(x_test)


# In[ ]:


# Generate Submission File 
results_df = pd.DataFrame({ 'PassengerId': test_df['PassengerId'], 'Survived': predictions })
results_df.to_csv("submission.csv", index=False)


# In[ ]:




