#!/usr/bin/env python
# coding: utf-8

# # XGBOOST + BAYESIAN SEARCH CV

# ## Load data
# 
# First we will be loading the titanic training dataset.
# 
# _Note the sample function, usefull in case the dataset is too big and you want to get only a sample from it. With the frac parameter yo define % of data to keep._

# In[21]:


#LOAD DATA
import pandas as pd
X = pd.read_csv('../input/train.csv')
# X = X.sample(frac = 0.5, replace=False, random_state=2019)
X.head()


# ## Feature Engineering
# 
# For now we will just drop features that can't be used as is, and just binnarize the categorical features to use them in the model.
# 
# In future updates I will be adding some feature engineering.

# In[ ]:


#FEATURE TRANSFORMATION
import pandas as pd

y = X.Survived

X = X.drop(['PassengerId','Survived','Name','Ticket','Cabin'], axis=1)
X = pd.get_dummies(X)

X.head()


# ## Train the model

# ### Bayesian Search for finding the best Hyperparameters
# 
# First we will use bayesian search on 50% of the data to find the best Hyperparameters and then with those parameters we will train the final model with 100% of the train data.

# In[ ]:


# TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split

X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.5, random_state=2019)


# In[ ]:


# #TRAIN MODEL - BAYESIAN SEARCH
import xgboost as xgb
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold

bayes_cv_tuner = BayesSearchCV(estimator = xgb.XGBClassifier(
                                    n_jobs = -1,
                                    objective = 'binary:logistic',
                                    eval_metric = 'auc',
                                    silent=1,
                                    early_stopping = 200,
                                    tree_method='approx'),
                search_spaces = {
                    'min_child_weight': (1, 50),
                    'max_depth': (3, 10),
                    'max_delta_step': (0, 20),
                    'subsample': (0.01, 1.0, 'uniform'),
                    'colsample_bytree': (0.01, 1.0, 'uniform'),
                    'colsample_bylevel': (0.01, 1.0, 'uniform'),
                    'reg_lambda': (1e-2, 1000, 'log-uniform'),
                    'reg_alpha': (1e-2, 1.0, 'log-uniform'),
                    'gamma': (1e-2, 0.5, 'log-uniform'),
                    'min_child_weight': (0, 20),
                    'scale_pos_weight': (1e-6, 500, 'log-uniform'),
                    'n_estimators': (150,1000),
                    'learning_rate':(0.01,0.08,'uniform'),
                    'subsample':(0.01,1,'uniform'),
                    'eta':(0.01,0.2,'uniform')
                },    
                scoring = 'roc_auc',
                cv = StratifiedKFold(
                    n_splits=5,
                    shuffle=True,
                    random_state=42),
                n_jobs = -1,
                n_iter = 10,   
                verbose = 1,
                refit = True,
                random_state = 786)

tunning_model = bayes_cv_tuner.fit(X_train, y_train)


# In[ ]:


# TRAINING BEST MODEL RESULTS
print('BEST ESTIMATOR: '+ str(tunning_model.best_estimator_))
print('BEST SCORE: '+ str(tunning_model.best_score_))
print('BEST PARAMS: '+ str(tunning_model.best_params_))


# In[ ]:


#TEST MODEL
y_pred = tunning_model.predict_proba(X_eval)

# roc_auc score
from sklearn.metrics import roc_auc_score
print('ROC AUC SCORE ON TESTING DATA: '+str(roc_auc_score(y_eval,y_pred[:,1])))


# In[ ]:


# FIND BEST ACCURACY PROB
import numpy as np
from sklearn.metrics import accuracy_score

best_acc = 0
best_prob = 0

for i in np.arange(0,1,0.01):
    y_pred_tmp = np.where((y_pred[:,1] >= i),1,0)
    acc_tmp = accuracy_score(y_eval, y_pred_tmp) 
    if acc_tmp > best_acc:
        best_prob = i
        best_acc = acc_tmp
        
print('Model Best Accracy: {0} - with Prob: {1}'.format(best_acc,best_prob))


# ### Train final model
# 
# We will use the hyperparameters found + 100% of the data.

# In[ ]:


# TRAIN FINAL MODEL
import xgboost as xgb

final_model = xgb.XGBClassifier(**tunning_model.best_params_,
                                n_jobs = -1,
                                objective = 'binary:logistic',
                                eval_metric = 'auc',
                                silent=1,
                                early_stopping = 200,
                                tree_method='approx')
final_model.fit(X, y)


# In[ ]:


#SAVE MODEL
from sklearn.externals import joblib
joblib.dump(final_model, 'model_xgboost.pkl')


# ## Create submission

# In[ ]:


# LOAD SUBMIT DATA
X_submit = pd.read_csv('../input/test.csv')
X_submit.head()


# In[ ]:


# FEATURE TRANSFORMATION
import pandas as pd

PassengerId = X_submit.PassengerId
X_submit = X_submit.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
X_submit = pd.get_dummies(X_submit)

X_submit.head()


# In[ ]:


# PREDICT
y_submit = final_model.predict_proba(X_submit)


# In[ ]:


# CONVERT PROBABILITIES PREDICTED TO Survived = 0 / 1
import numpy as np
y_submit_final = np.where((y_submit[:,1] >= best_prob),1,0)


# In[ ]:


#SAVE SUBMISSION
submission = pd.DataFrame({'PassengerId':PassengerId,
                           'Survived':y_submit_final})
submission.to_csv('submission_xgboost.csv',index=False)

