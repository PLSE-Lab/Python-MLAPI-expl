#!/usr/bin/env python
# coding: utf-8

# # Santander 
# 
# This is a very simple kernel for the Santander competition featuring the following:
# * interaction features based on those features that show correlations with `target`;
# * re-sampling of the data to compensate for the imbalanced nature of the training data set using `imblearn`;
# * comparison of and modeling with `lightgbm` (using the `sklearn` interface) and a simple ridge classifier;
# * use of `GridSearchCV` and `sklearn.pipeline` to find the best-fit parameters;
# * no magic.
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# # Data Exploration

# ## Feature completeness and data types

# In[2]:


train.info(max_cols=250)


# In[3]:


test.info(max_cols=250)


# Inventory:
# * `train`: 200 features with continuous data + binary `target` feature + `ID_code` column; all features complete, no missing data
# * `test`: 200 features with continuous data + `ID_code` column; all features complete, no missing data
# 
# Feature labels are **anonymized**.

# What is the distribution of the data in each feature?

# In[4]:


f, ax = plt.subplots(figsize=(25,5))

train.drop(['target', 'ID_code'], axis=1).plot.box(ax=ax, rot=90)


# In general, all features are **well-behaved**, most box-plot bars are symmetric, so **distributions are more or less symmetric**, as well. 

# Is the training data set balanced?

# In[5]:


len(train.loc[train.target == 1])/len(train)


# No, only about 10% of the training set have `target == 1` - the training **data set is highly imbalanced**. Will have to use resampling in the modeling!

# ## Correlation analysis
# 
# Check for correlations between individual features:

# In[6]:


r2 = pd.concat([train.drop(['target', 'ID_code'], axis=1), test.drop('ID_code', axis=1)]).corr()**2
r2 = np.tril(r2, k=-1)  # remove upper triangle and diagonal
r2[r2 == 0] = np.nan # replace 0 with nan


# In[7]:


f, ax = plt.subplots(figsize=(20,20))
sns.heatmap(np.sqrt(r2), annot=False,cmap='viridis', ax=ax)


# Explained **absolute variation ($\sqrt{r^2}$) between individual features is small** (< 1%). All features seem to be highly independent from each other.
# 
# What about correlations between features and `target`?

# In[8]:


target_r2 = train.drop(['ID_code', 'target'], axis=1).corrwith(train.target).agg('square')

f, ax = plt.subplots(figsize=(25,5))
target_r2.agg('sqrt').plot.bar(ax=ax)


# Explained absolute variation ($\sqrt{r^2}$) up to 8%. Few features seem to stick out in terms of correlation with `target`.

# ## Feature Engineering
# 
# Not much is known about the features, except for what is listed above. This makes feature engineering hard. In a long (and blind) shot, we try the following:
# 
# Extract the top $n$ features with $\sqrt{r^2} \geq 0.048$ and create second-degree interaction features based on those. These polynomial features will be added in the modeling step.
# 
# The value of 0.048 was derived by maximizing the `roc-auc` for the ridge regression model below. Interestingly, adding these interaction features to the `lightgbm` model does not improve its predictive power. 

# In[9]:


top = target_r2.loc[np.sqrt(target_r2) > 0.048].index
top


# In[10]:


from sklearn.preprocessing import PolynomialFeatures

polyfeat_train = pd.DataFrame(PolynomialFeatures(2).fit_transform(train[top]))
polyfeat_test = pd.DataFrame(PolynomialFeatures(2).fit_transform(test[top]))


# # Modeling
# 
# We use `lightgbm` and a simple ridge regression as representatives of tree-based and linear models, respectively.
# 
# Given the highly imbalanced nature of the training data set, we apply a resampler for balancing. We tested `RandomUnderSampler` and `RandomOverSampler` from the `imblearn` module and found that the latter performs better with both models.**

# In[11]:


from imblearn.over_sampling import RandomOverSampler


# In[12]:


# additional imports
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler


# ## lightgbm

# In[33]:


from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb

lgbpipe = Pipeline([('resample', RandomOverSampler(random_state=42)), ('model', lgb.LGBMClassifier(random_state=42, objective='binary', metric='auc', 
                                                                                                   boosting='gbdt', verbosity=1,
                                                                                                   tree_learner='serial'))])

params = {    
    "model__max_depth" : [20],
    "model__num_leaves" : [30],
    "model__learning_rate" : [0.1],
    "model__subsample_freq": [5],
    "model__subsample" : [0.3],
    "model__colsample_bytree" : [0.05],
    "model__min_child_samples": [100],
    "model__min_child_weight": [10],
    "model__reg_alpha" : [0.12],
    "model__reg_lambda" : [15.5],
    "model__n_estimators" : [600]
    }

# previous best-fit gridsearch parameters and results
# {'model__colsample_bytree': 0.05, 'model__learning_rate': 0.1, 'model__max_depth': 20, 'model__min_child_samples': 100, 'model__min_child_weight': 10, 'model__n_estimators': 100, 'model__num_leaves': 30, 'model__reg_alpha': 0.1, 'model__reg_lambda': 10, 'model__subsample': 0.3, 'model__subsample_freq': 5}
# 0.8735588789424164
# {'model__colsample_bytree': 0.05, 'model__learning_rate': 0.1, 'model__max_depth': 20, 'model__min_child_samples': 100, 'model__min_child_weight': 10, 'model__n_estimators': 400, 'model__num_leaves': 30, 'model__reg_alpha': 0.12, 'model__reg_lambda': 0.2, 'model__subsample': 0.3, 'model__subsample_freq': 5}
# 0.8915905852982839
# {'model__colsample_bytree': 0.05, 'model__learning_rate': 0.1, 'model__max_depth': 20, 'model__min_child_samples': 100, 'model__min_child_weight': 10, 'model__n_estimators': 500, 'model__num_leaves': 30, 'model__reg_alpha': 0.12, 'model__reg_lambda': 0.2, 'model__subsample': 0.3, 'model__subsample_freq': 5}
# 0.8923071245054173
# {'model__colsample_bytree': 0.05, 'model__learning_rate': 0.1, 'model__max_depth': 20, 'model__min_child_samples': 100, 'model__min_child_weight': 10, 'model__n_estimators': 600, 'model__num_leaves': 30, 'model__reg_alpha': 0.12, 'model__reg_lambda': 0.2, 'model__subsample': 0.3, 'model__subsample_freq': 5}
# 0.8925518240005254
# {'model__colsample_bytree': 0.05, 'model__learning_rate': 0.1, 'model__max_depth': 20, 'model__min_child_samples': 100, 'model__min_child_weight': 10, 'model__n_estimators': 550, 'model__num_leaves': 30, 'model__reg_alpha': 0.12, 'model__reg_lambda': 0.2, 'model__subsample': 0.3, 'model__subsample_freq': 5}
# 0.8924978701504809
# {'model__colsample_bytree': 0.05, 'model__learning_rate': 0.1, 'model__max_depth': 20, 'model__min_child_samples': 100, 'model__min_child_weight': 10, 'model__n_estimators': 600, 'model__num_leaves': 30, 'model__reg_alpha': 0.12, 'model__reg_lambda': 15, 'model__subsample': 0.3, 'model__subsample_freq': 5}
# 0.8941148812638564
# {'model__colsample_bytree': 0.05, 'model__learning_rate': 0.1, 'model__max_depth': 20, 'model__min_child_samples': 100, 'model__min_child_weight': 10, 'model__n_estimators': 600, 'model__num_leaves': 30, 'model__reg_alpha': 0.5, 'model__reg_lambda': 12, 'model__subsample': 0.3, 'model__subsample_freq': 5}
# 0.8938169988416745
# {'model__colsample_bytree': 0.05, 'model__learning_rate': 0.1, 'model__max_depth': 20, 'model__min_child_samples': 100, 'model__min_child_weight': 10, 'model__n_estimators': 600, 'model__num_leaves': 30, 'model__reg_alpha': 0.3, 'model__reg_lambda': 15, 'model__subsample': 0.3, 'model__subsample_freq': 5}
# 0.8941407236592286
# {'model__colsample_bytree': 0.05, 'model__learning_rate': 0.1, 'model__max_depth': 20, 'model__min_child_samples': 100, 'model__min_child_weight': 10, 'model__n_estimators': 600, 'model__num_leaves': 30, 'model__reg_alpha': 0.2, 'model__reg_lambda': 15, 'model__subsample': 0.3, 'model__subsample_freq': 5}
# 0.8938875270813017
# {'model__colsample_bytree': 0.05, 'model__learning_rate': 0.1, 'model__max_depth': 20, 'model__min_child_samples': 100, 'model__min_child_weight': 10, 'model__n_estimators': 600, 'model__num_leaves': 30, 'model__reg_alpha': 0.2, 'model__reg_lambda': 15, 'model__subsample': 0.3, 'model__subsample_freq': 5}
# 0.8938875270813017
# {'model__colsample_bytree': 0.05, 'model__learning_rate': 0.1, 'model__max_depth': 20, 'model__min_child_samples': 100, 'model__min_child_weight': 10, 'model__n_estimators': 600, 'model__num_leaves': 30, 'model__reg_alpha': 0.12, 'model__reg_lambda': 15.5, 'model__subsample': 0.3, 'model__subsample_freq': 5}
# 0.8943001048082946
# {'model__colsample_bytree': 0.05, 'model__learning_rate': 0.1, 'model__max_depth': 20, 'model__min_child_samples': 100, 'model__min_child_weight': 10, 'model__n_estimators': 600, 'model__num_leaves': 30, 'model__reg_alpha': 0.12, 'model__reg_lambda': 15.2, 'model__subsample': 0.3, 'model__subsample_freq': 5}
# 0.8939732044413886

lgbgrid = GridSearchCV(lgbpipe, param_grid=params, cv=10, scoring='roc_auc')
lgbgrid.fit(train.drop(['ID_code', 'target'], axis=1), train.target)

print(lgbgrid.best_params_)
print(lgbgrid.best_score_)


# ## Ridge regression
# 
# Using a `RobustScaler`, which is probably not even necessary since the features are rather well-behaved.

# In[14]:


from sklearn.linear_model import RidgeClassifier

ridgepipe = Pipeline([('resample', RandomOverSampler(random_state=42)), ('scaler', RobustScaler()), ('model', RidgeClassifier(random_state=42))])

params = {'model__alpha': [1.0]} # between 0.5 and 2; best-fit so far: 1
 
ridgegrid = GridSearchCV(ridgepipe, param_grid=params, cv=3, scoring='roc_auc')
ridgegrid.fit(pd.concat([train.drop(['ID_code', 'target'], axis=1), polyfeat_train], axis=1, join='inner'), train.target)

print(ridgegrid.best_params_)
print(ridgegrid.best_score_)


# Comparing the results between `lightgbm` and the ridge regression classifier, the former clearly wins despite the additional interaction features that are used in the ridge regression model. 

# 

# # Submission

# In[15]:


pred = pd.DataFrame(lgbgrid.predict_proba(test.drop(['ID_code'], axis=1))[:, -1], columns=['target'], index=test.loc[:, 'ID_code'])
pred.to_csv('submission.csv', index=True)


# In[16]:


get_ipython().system('head submission.csv')


# In[17]:


test.head()


# # Changelog
# 1. initial commit: cv score: 0.89, public score: 0.80
# 2. explored a larger parameter space for `lightgbm` and force higher regularization parameters to prevent overfitting; submit probabilities instead of classes: cv score: 0.873, public score: 0.870
# 3. some more fine-tuning of the model: cv score: 0.894 -- final submission
