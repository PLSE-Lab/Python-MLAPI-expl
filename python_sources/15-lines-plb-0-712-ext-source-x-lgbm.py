#!/usr/bin/env python
# coding: utf-8

# # A quick and simple GB model optimisation on EXT\_SOURCE\_\* variables
# This kernel has started from the simple and clear [15 lines: Just EXT_SOURCE_x](https://www.kaggle.com/lemonkoala/15-lines-just-ext-source-x) by [Lem Lordje Ko](https://www.kaggle.com/lemonkoala). Goal goal is to see what performance can one reach in short piece of code. What has been added on top on the original kernel is optimisation of LightGBM hyper-parameters. The final reported precision is 0.723 locally and 0.712 on the public leaderboard

# In[ ]:


import pandas as pd 
import numpy as np
import lightgbm as lgb

data = pd.read_csv("../input/application_train.csv")
test = pd.read_csv("../input/application_test.csv")


# Define parameter range in which optimisation will be performed.

# In[ ]:


from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
param_test ={'num_leaves': sp_randint(6, 50), 
             'min_child_weight': sp_randint(1, 500), 
             'colsample_bytree': sp_uniform(loc=0.6, scale=0.4), 
             'subsample': sp_uniform(loc=0.2, scale=0.8), 
             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}


# Define the hyper-parameter optimiser, it will test `n_HP_points_to_test` points sampled randomly. Beware: 3x20 (`CV_folds x n_HP_points_to_test`)  will run for approx 3 min on 4 CPU cores on kaggle

# In[ ]:


n_HP_points_to_test = 20
from sklearn.model_selection import RandomizedSearchCV
clf = lgb.LGBMClassifier(max_depth=-1, is_unbalance=True, random_state=314, silent=True, metric='None', n_jobs=5)
gs = RandomizedSearchCV(
    estimator=clf, param_distributions=param_test, 
    n_iter=n_HP_points_to_test,
    scoring='roc_auc',
    cv=5,
    refit=True,
    random_state=314,
    verbose=True)


# Do actual parameter tune

# In[ ]:


gs.fit(data.filter(regex=r'^EXT_SOURCE_.', axis=1), data['TARGET'])
print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))


# Let's print the 5 best parameter sets based on the average roc auc on the testing fold in CV

# In[ ]:


print("Valid+-Std     Train  :   Parameters")
for i in np.argsort(gs.cv_results_['mean_test_score'])[-5:]:
    print('{1:.4f}+-{3:.4f} {2:.4f}   :  {0}'.format(gs.cv_results_['params'][i], 
                                    gs.cv_results_['mean_test_score'][i], 
                                    gs.cv_results_['mean_train_score'][i],
                                    gs.cv_results_['std_test_score'][i]))


# Prepare a submission (note that you can directly submit it from the `Output` tab of the kernel, when you fork it)

# In[ ]:


probabilities = gs.best_estimator_.predict_proba(test.filter(regex=r'^EXT_SOURCE_.', axis=1))
submission = pd.DataFrame({
    'SK_ID_CURR': test['SK_ID_CURR'],
    'TARGET':     [ row[1] for row in probabilities]
})
submission.to_csv("submission.csv", index=False)


# In[ ]:




