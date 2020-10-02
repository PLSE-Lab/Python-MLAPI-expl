#!/usr/bin/env python
# coding: utf-8

# # About this kernel
# 
# #### I go through various standard techniques using XGBoost using GPU processor, which naturally is much faster than CPU.
# 
# This Kernel uses much material from other kernels, namely 
# https://www.kaggle.com/xhlulu/ieee-fraud-xgboost-with-gpu-fit-in-40s by @xhlulu 
# https://www.kaggle.com/babatee/intro-xgboost-classification  by @babatee
# https://www.kaggle.com/vinhnguyen/accelerating-hyper-parameter-searching-with-gpu by Vinh Nguyen
# 
# ### Please enjoy and upvote!
# 
# 

# In[ ]:


import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split ,GridSearchCV,StratifiedKFold,RandomizedSearchCV
from sklearn.metrics import mean_absolute_error , accuracy_score ,confusion_matrix
print("XGBoost version:", xgb.__version__)


# Preprocessing the data (reading and splitting for training and validation)

# In[ ]:


#get the data
test = pd.read_csv('../input/learn-together/test.csv')
train = pd.read_csv('../input/learn-together/train.csv')

#prepare the data fir training and Validation

Y = train['Cover_Type']
X = train.copy()
X.drop(columns = ['Cover_Type'] , inplace=True )
x_train , x_val , y_train , y_val = train_test_split(X ,Y ,  train_size=0.8, test_size=0.2, 
                                                      random_state=0)


# # Training
# 
# #### To activate GPU usage, simply use `tree_method='gpu_hist'. We set a standard model up first. 

# In[ ]:



get_ipython().run_cell_magic('time', '', 'model = xgb.XGBClassifier(\nlearning_rate =0.1,\nn_estimators=500,\nmax_depth=5,\nmin_child_weight=1,\ngamma=0,\nsubsample=0.8,\ncolsample_bytree=0.8,\nobjective= \'multi:softmax\',\nnthread=9,\nscale_pos_weight=1,\nseed=27,\ntree_method=\'gpu_hist\' )\n\n#training\ntrain_model = model.fit(x_train, y_train)\npred = train_model.predict(x_val)\nprint("Accuracy for model 3: %.2f" % (accuracy_score(y_val, pred) * 100))')


# In[ ]:


#Get classification_report ,   
from sklearn.metrics import classification_report
target_names = ['Spruce/Fir' , 'Lodgepole Pine','Ponderosa Pine','Cottonwood/Willow','Aspen','Douglas-fir','Krummholz']
print( (classification_report(y_val, pred,target_names=target_names)))


# Model  has a pretty decent performance. However, model seem to struggle predicting Spruce/Fir and Lodgepole Pine. Models do great on Cottonwood/Willow and Krummholz. Let's look at the confusion matrix:

# In[ ]:


#Confusion matrix
confusion_matrix(y_val, pred)


# For Spruce/Fir and Lodgepole Pine, there seem to be many false predictions. An attempt needs to be made for imporvements on these two categories.

# Let's do a little  Hyperparameter Tunning. 

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n\nfolds = 2\nparam_comb = 20\nparams = {\n        \'min_child_weight\': [1, 5, 10],\n        \'gamma\': [0.5, 1, 1.5, 2, 5],\n        \'subsample\': [0.6, 0.8, 1.0],\n        \'colsample_bytree\': [0.6, 0.8, 1.0],\n        \'max_depth\': [3, 5, 7, 10],\n        \'learning_rate\': [0.01, 0.02, 0.05]    \n        }\n\nskf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)\n\nxgbb = XGBClassifier(learning_rate=0.01, n_estimators=250, objective=\'binary:logistic\',\n                    silent=True, nthread=6, tree_method=\'gpu_hist\', eval_metric=\'auc\')\n\nrandom_search = RandomizedSearchCV(xgbb, param_distributions=params, n_iter=param_comb, n_jobs=4, cv=skf.split(x_train,y_train), verbose=3, random_state=1001 )\n\ntrain_model =random_search.fit(x_train,y_train)\npred = train_model.predict(x_val)\n\nprint("Accuracy for model is: %.2f" % (accuracy_score(y_val, pred) * 100))\nprint(train_model.best_params_)\nprint(train_model.best_estimator_)\ncvres = train_model.cv_results_\n\nfor mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):\n    print(mean_score, params)')


# Now that we have good hyperparameters, we train a new model:

# In[ ]:


model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1.0, eval_metric='auc',
              gamma=2, learning_rate=0.05, max_delta_step=0, max_depth=10,
              min_child_weight=1, missing=None, n_estimators=250, n_jobs=1,
              nthread=6, objective='multi:softprob', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=True, subsample=0.8, tree_method='gpu_hist', verbosity=1)

#training
train_model = model.fit(x_train, y_train)
pred = train_model.predict(x_val)
print("Accuracy for model 3: %.2f" % (accuracy_score(y_val, pred) * 100))


# #### Model was only slightly improved to achieve an accuracy score of 87.53. This is about the extent that XGboost can be improved without making adjustments to the data/attributes (scaling, encoding, etc).

# In[ ]:



submit = pd.DataFrame(train_model.predict(test) , columns = [ 'Cover_Type'])
submit['ID']=test['Id']
submit.to_csv('submission.csv' , index=False)

