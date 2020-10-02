#!/usr/bin/env python
# coding: utf-8

# # Baseline Prediction accuracy with Zero Feature Selection
# ## Hyper parameter optimization with Tree based algorithms
# In this Kernel, we will focus on finding the best classification model for each of **Random Forest, Extremely Randomized Trees and LightGBM classifiers**. We will use **all features as is**, to get a feel for the ability of these models, given an optimal set of hyperparameters, to produce a reasonably predicting model with **absolutely no feature selection performed**. 
# 
# The reason i chose Tree based algorithms to baseline prediction accuracy is these algorithms need no pre processing, are resistant to outliers and have historically shown best performance in prediction competitions. 
# 
# While Random Forests are valued for their ability to decorrelate the individual decision trees by randomizing the split variables that are chosen, Extra Trees take this randomization one step further by choosing even the split points at random thereby decorrelating the trees to a even greater extent. This gives very diverse trees which can potentially perform even better with unseen test data as they further reduce overfitting (hopefully not underfit as much). 
# 
# I used LGBM to apply boosting which basically sequentially improves predictions by focussing on the error generated at each tree. I used it for its ability to run faster on large datasets. Our train set may be small with only 15000 rows but the test set has 0.5  million rows. LGBM is different from other boosting algorithms in the way the trees are grown, but i am not getting into details here. This [link](http://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc) is kind of a primer to knowing LGBM before checking out official documentation
# 
# Once we have determined the best choice of parameters for each of these algorithms,we will apply these parameters to predict and baseline accuracy levels.
# 
# Finally we will also try to see if a stacked classifier with these 3 models as inputs is able to come up with predictions as good or better than these individual models.
# 
# The final best model for these three tree algorithms and the stacked classifier will be executed in the next kernel so that we do not end up running hyper parameter optimization again and consuming computing resources unnessarily. All the heavy lifting will be done in this Kernel. I want to publish even the random grid models logs to record how the model fitting process took place and analyze the metrics as against each set of hyperparameters.
# 
# We will use **RandomizedSearchCV instead of GridSearchCV** as the latter will take up too much computing resources if taking a wider range of values for each hyperparameter. I will set number of iteration as 100 with 3 folds of Cross Validation so that we are actually fitting 300 models, which can be considered a decent number of experiments to find an optimal set of hyper parameters.
# 
# Finally, We will show 2 metrics for each model built- **Accuracy as well as Negative Log loss** and use Negative log loss as the basis for choosing the best model. This is because, log loss accords severe penalty for generating a higher probability of class prediction and then getting the class prediction wrong during the model training.

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


# Parameters for Random Forest Hyperparameter tuning
param_grid_rf = {'n_estimators': np.logspace(2,3.5,8).astype(int),
                 'max_features': [0.1,0.3,0.5,0.7,0.9],
                 'max_depth': np.logspace(0,3,10).astype(int),
                 'min_samples_split': [2, 5, 10],
                 'min_samples_leaf': [1, 2, 4],
                 'bootstrap':[True, False]}

# Instantiate RandomForestClassifier
rf = RandomForestClassifier(random_state = 42)

# Create a Random Search Parameter Grid
grid_rf = RandomizedSearchCV(estimator=rf, 
                          param_distributions=param_grid_rf, 
                          n_iter=100,
                          cv=3, 
                          verbose=3, 
                          n_jobs=1,
                          scoring = {'NLL':'neg_log_loss', 'Accuracy':'accuracy'}, 
                          refit='NLL')
# Fit Random Forest Models - 100 models each with 3 rounds of cross validation - 300 fitted models
grid_rf.fit(X,y)
print('The Best Random Forest Estimator is: ', grid_rf.best_estimator_)
print('The Best Random Forest Parameters are: ', grid_rf.best_params_)
print('The Best Random Forest score is: ', grid_rf.best_score_)


# In[ ]:


# Parameters for Extremely Randomized Trees Hyperparameter tuning
param_grid_extra = {'n_estimators': np.logspace(2,3.5,8).astype(int),
                    'max_features': [0.1,0.3,0.5,0.7,0.9],
                    'max_depth': np.logspace(0,3,10).astype(int),
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'bootstrap':[True, False]}

# Instantiate ExtraTreesClassifier
extra_trees = ExtraTreesClassifier(random_state = 42)

# Create a Random Search Parameter Grid
grid_extra_trees = RandomizedSearchCV(estimator=extra_trees, 
                          param_distributions=param_grid_extra, 
                          n_iter=100,
                          cv=3, 
                          verbose=3, 
                          n_jobs=1,
                          scoring = {'NLL':'neg_log_loss', 'Accuracy':'accuracy'}, 
                          refit='NLL')
# Fit Extra Trees Models - 100 models each with 3 rounds of cross validation - 300 fitted models
grid_extra_trees.fit(X,y)
print('The Best Extra Trees Estimator is: ', grid_extra_trees.best_estimator_)
print('The Best Extra Trees Parameters are: ', grid_extra_trees.best_params_)
print('The Best Extra Trees score is: ', grid_extra_trees.best_score_)


# In[ ]:


# Parameters for Light Gradient Boosting Machines Hyperparameter tuning
param_grid_lgbm = {'n_estimators': np.logspace(2,3.5,8).astype(int),
                   'feature_fraction': [0.1,0.3,0.5,0.7,0.9],
                   'bagging_fraction': [0.5,0.6,0.7,0.8,0.9],
                   'max_depth': np.logspace(0,3,10).astype(int),
                   'min_samples_split': [2, 5, 10],
                   'min_data_in_leaf': [1, 2, 4],
                   'learning_rate':[0.005,0.01,0.05,0.1,0.5]}

# Instantiate ExtraTreesClassifier
lgbm = LGBMClassifier(random_state = 42, is_provide_training_metric = True)

# Create a Random Search Parameter Grid
grid_lgbm = RandomizedSearchCV(estimator=lgbm, 
                          param_distributions=param_grid_lgbm, 
                          n_iter=100,
                          cv=3, 
                          verbose=3, 
                          n_jobs=1,
                          scoring = {'NLL':'neg_log_loss', 'Accuracy':'accuracy'}, 
                          refit='NLL')
# Fit LightGBM Models - 100 models each with 3 rounds of cross validation - 300 fitted models
grid_lgbm.fit(X,y)
print('The Best LightGBM Estimator is: ', grid_lgbm.best_estimator_)
print('The Best LightGBM Parameters are: ', grid_lgbm.best_params_)
print('The Best LightGBM score is: ', grid_lgbm.best_score_)


# ## **Up Next:**
# 
# In a separate kernel, Plug in the best parameters for each of the above 3 classifiers as well as construct a stacked classifier on top of these classifiers.

# Create Stacked Classifier using Random Forest, Extra Trees and LGBM. Use Extra Trees as meta classifier
# stack = StackingCVClassifier(classifiers=[rf,
#                                          extra_trees,
#                                          lgbm],
#                             use_probas=True,
#                             meta_classifier=extra_trees, random_state = 42)
# cross_val_score(stack,X,y,cv = 5, scoring = 'accuracy', verbose = 3)
