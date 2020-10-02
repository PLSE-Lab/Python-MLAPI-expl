#!/usr/bin/env python
# coding: utf-8

# # Objective of this Kernel: 
# 
# We earlier performed hyperparameter tuning for Random Forest, Extremely Randomized Trees and LightGBM classifiers in my previous kernel -  [ForestML: All features - Tree Algo HP Optimization](http://www.kaggle.com/aravindankrishnan/forestml-all-features-tree-algo-hp-optimization). We did this for the training set without any feature selection to generate a baseline accuracy view before applying any feature selection measures. We will now just plug in the best estimators and check out our validation set accuracy for these 3 classifiers.
# 
# Once we have done that, we will also fit a stacking classifier that is an ensemble of the best estimators of these 3 classifiers (*ensemble of ensembles!!*) and check out its performance with respect to the individual best estimators.
# 

# Since this is a new kernel, we need to run some code for getting the training data in place for ML.

# # Training Data Preparation

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


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from mlxtend.classifier import StackingCVClassifier
from sklearn.model_selection import cross_validate


# In[ ]:


import pandas as pd
train = pd.read_csv('../input/learn-together/train.csv', index_col = 'Id')
test = pd.read_csv('../input/learn-together/test.csv', index_col = 'Id')
train.columns
train.info()
train.head()


# In[ ]:


# Make a copy of train df for ML experiments
train_2 = train.copy()
train_2.columns


# In[ ]:


# Separate feature and target arrays as X and y
X = train_2.drop('Cover_Type', axis = 1)
y=train_2.Cover_Type
print(X.columns)
y[:5]


# In[ ]:


# Split X and y into Train and Validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X,y,test_size = 0.2,random_state = 99)


# Plug in the best hyper parameters for Random Forest, Extra Trees and LightGBM found in my previous kernel - [ForestML: All features - Tree Algo HP Optimization](http://www.kaggle.com/aravindankrishnan/forestml-all-features-tree-algo-hp-optimization)

# # Random Forest Best Estimator 

# In[ ]:


# Plug in Random Forest Best Estimator Parameters
rf = RandomForestClassifier(n_estimators = 1930, 
                            min_samples_split = 5, 
                            min_samples_leaf = 1, 
                            max_features = 0.3, 
                            max_depth = 46, 
                            bootstrap = False,
                            random_state=42)
rf.fit(X_train,y_train)
y_pred_rf = rf.predict(X_val)
print('Random Forest Best Estimator validation set accuracy is: ', accuracy_score(y_val,y_pred_rf))


# # Extra Trees Best Estimator

# In[ ]:


# Plug in Extra Trees Best Estimator Parameters
extra_trees = ExtraTreesClassifier(n_estimators = 3162,
                                   min_samples_split = 5, 
                                   min_samples_leaf = 1, 
                                   max_features = 0.5,
                                   max_depth = 464, 
                                   bootstrap = False,
                                   random_state=42)
extra_trees.fit(X_train,y_train)
y_pred_extra = extra_trees.predict(X_val)
print('Extra Trees Best Estimator validation set accuracy is: ', accuracy_score(y_val,y_pred_extra))


# # LightGBM Best Estimator

# In[ ]:


# Plug in LightGBM Best Estimator Parameters
lgbm = LGBMClassifier(random_state = 42,
                      n_estimators = 268, 
                      min_samples_split = 5, 
                      min_data_in_leaf = 1, 
                      max_depth = 21, 
                      learning_rate = 0.05, 
                      feature_fraction = 0.5, 
                      bagging_fraction = 0.5 , 
                      is_training_metric = True)
lgbm.fit(X_train,y_train)
y_pred_lgbm = lgbm.predict(X_val)
print('LightGBM Best Estimator validation set accuracy is: ', accuracy_score(y_val,y_pred_lgbm))


# I got a number of warnings during LGBM execution that i left num_leaves parameter default value while my max_depth was way larger such that 2^maxdepth was way larger than num_leaves. So my accuracy was going to go for a toss. For LGBM Classifier, num_leaves is a more sensible parameter since trees are grown by leaves first than by level as is the case with other boosting models. The max_depth is only to avoid over fitting as the leaves can go very deep in a lightGBM model. One option is to choose a different max_depth grid and set num_leaves to a pretty large value so that num_leaves is able to optimize for the highest accuracy, which means rerunning the HP optimization for LGBM, which will be atleast 2 hours.
# 
# For now, i am going with a simple hack. From the fitted models, Choose the best combination of accuracy and NLL one that has much more number of estimators so that we can be a little more confident of accuracy. So i just manually analyze the 300 fitted models' estimator, accuracy and NLL and pick out a candidate 'best' estimator manually.

# In[ ]:


# Plug in LightGBM 'Manual' Potential Best Estimator parameters
lgbm = LGBMClassifier(random_state=42,
                      n_estimators=3162, 
                      min_samples_split=5, 
                      min_data_in_leaf=1, 
                      max_depth=21, 
                      learning_rate=0.05,
                      feature_fraction=0.9, 
                      bagging_fraction=0.6 , 
                      is_training_metric = True)
lgbm.fit(X_train,y_train)
y_pred_lgbm = lgbm.predict(X_val)
print('LightGBM Best Estimator validation set accuracy is: ', accuracy_score(y_val,y_pred_lgbm))


# Wow! This manual pick seemed a decent idea. However, for future purposes, i will try to pick different grids with Random Search and with much less n_iter so that i can 'sample test' the param grids. once i feel i have the right grid, go in for our 300 model execution marathon run (overnight, if needed!!)

# # Stacking Classifier

# ## Stacking Classifier - Combine best estimators of Random Forest, Extra Trees and LightGBM
# We will now build a Stacking classifier based on the above 3 classifiers and see if it outperforms all or some of the individual estimators. We will use StackingCVClassifier which also performs cross validation to ensure robustness of the model predictions. For a quick reading, You can go through [StackingCVClassifier ](http://rasbt.github.io/mlxtend/user_guide/classifier/StackingCVClassifier/). We need to specify a meta classifier that will combine the individual classifiers. We can use any model here as the meta classifier, not just the classifiers we are using in the stack.
# 
# I will plan to use extra trees for 2 reasons
# 
# 1. Faster execution as even split points are random, so much faster execution
# 2. If we want to predict unseen sets, we should probably have more diverse trees which Extra trees can build on account of its randomized execution.
# 
# I will perform 5 fold CV on the stack classifer to get a grasp on the training performance.

# In[ ]:


stack = StackingCVClassifier(classifiers=[rf,
                                         extra_trees,
                                         lgbm],
                            use_probas=True,
                            meta_classifier=extra_trees, random_state = 42)

stack.fit(X_train,y_train)
print('Stacking Classifier Cross-Validation accuracy scores are: ',cross_val_score(stack,X_train,y_train, cv = 5))
y_pred_stack = stack.predict(X_val)
print('Stacking Classifier validation set accuracy is: ', accuracy_score(y_val,y_pred_stack))


# Wow..The stacking classifier seems to have done a bit better than the individual classifiers. We could repeat this with different other classifiers to see how they 'stack' up against other stacks!! For now, we have a reasonable baseline which seems to be decent enough to give us a tough threshold for feature selection. 

# # Feature Importances
# We will get the feature importances for the 3 individual Tree based best estimator classifiers

# ## Feature Importances from Random Forest Best Estimator

# In[ ]:


# Feature Importances from Random Forest Best Estimator
rf_importance = pd.DataFrame(list(zip(X_train.columns, list(rf.feature_importances_))), columns = ['Feature', 'Importance'])
rf_importance.sort_values('Importance', ascending = False)


# ## Feature Importances from Extra Trees Best Estimator

# In[ ]:


# Feature Importances from Extra Trees Best Estimator
extra_importance = pd.DataFrame(list(zip(X_train.columns, list(extra_trees.feature_importances_))), columns = ['Feature', 'Importance'])
extra_importance.sort_values('Importance', ascending = False)


# ## Feature Importances from LightGBM Best Estimator

# In[ ]:


# Feature Importances from LGBM Best Estimator
lgbm_importance = pd.DataFrame(list(zip(X_train.columns, list(lgbm.feature_importances_))), columns = ['Feature', 'Importance'])
lgbm_importance.sort_values('Importance', ascending = False)


# # Up Next:
# 
# We will experiment with different types of feature selection approaches and see how they provide us good ways to improve performance. For one of them based on feature importances, we can use the above feature importances to filter out some features and apply hyperparameter tuning for the remaining features. That will be for the next kernel.
