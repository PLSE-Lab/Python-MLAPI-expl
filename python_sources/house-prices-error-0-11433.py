#!/usr/bin/env python
# coding: utf-8

# ## House Prices: Advanced Regression Techniques
# 
# * Data preprocessing steps is based on this note book :[notebook](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard) so I haven't gone through the proprocessing steps in this notebook .
# * You also can find the preprcessed data and the simplified steps for preprocessing of the data in my github repository:[repository](https://github.com/Moeinh77/Kaggle-House-Prices-Advanced-Regression-Techniques)
# *  Hyper parameters have been found by GridSearch and randomizedSearch of scikit learn .
# * The final model is a weitghted  average of : a 3 layer stack ensemble , a LGboost  and a XGBoost model.
# * This kernel guides you through a smaller error than all the other kernels so far, because of multiple layers of stacking (3 models in first stack,3 models in the second stack and one model as an estimator in last stack).
# * For ease of use I have used this library for stacking : [vecstack](https://github.com/vecxoz/vecstack).
# * You can see how the stack used has been implemented in here :[implementation](https://github.com/vecxoz/vecstack/blob/master/examples/00_stacking_concept_pictures_code.ipynb).
# ---
# 
# ### Possible improvements:
# * Try different ways for preprocessing the data (e.g using only most importanant features of data for some models)
# * Increasing number of layers with more models in each layer 
# * Decreasing the corrolation between 3 models in final averaged ensemble

# ### Importing libraries

# In[ ]:


#import some necessary librairies
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLars,RidgeCV
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


# In[ ]:


import pandas as pd
train =pd.read_csv('../input/housingames/X_train.csv')
test = pd.read_csv('../input/housingames/X_test.csv')
ytrain=pd.read_csv('../input/housingames/y_train.csv')


# ### Metric function

# In[ ]:


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# ## Stacking

# ![Image](https://camo.githubusercontent.com/fa34150cb31d02f68886584d549f300f8c290ba3/68747470733a2f2f6769746875622e636f6d2f766563786f7a2f766563737461636b2f7261772f6d61737465722f7069632f616e696d6174696f6e322e676966)

# #### layer 1

# Tree based models do not need data to be scaled !
# so I haven't use scaling when predicting with boosting models

# In[ ]:


ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9,
                                                random_state=7))
#########################################################################
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

#########################################################################

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)


# In[ ]:


# library used for stacking 
get_ipython().system('pip install vecstack')


# In[ ]:


from vecstack import stacking

estimators = [KRR,GBoost,ENet]
X_train=train
y_train=ytrain
X_test=test
k=5

L_train_1, L_test_1=stacking(estimators,X_train,
         y_train, X_test,regression=True, 
         n_folds=k,mode='oof_pred',random_state=7, 
         verbose=2)


# #### layer 2

# In[ ]:


ENet2 = make_pipeline(RobustScaler(), ElasticNet(alpha=0.00055, l1_ratio=.45,
                                                random_state=7))
#########################################################################
KRR2 = KernelRidge(alpha=0.4, kernel='polynomial', degree=2, coef0=2.5)
#########################################################################
GBoost2 = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01,
                                   max_depth=3, max_features='sqrt',
                                   min_samples_leaf=7, min_samples_split=10, 
                                   loss='huber', random_state =7)


# In[ ]:


#layer 2
estimatorsL2=[ENet2,KRR2,GBoost2]

L_train_2, L_test_2=stacking(estimatorsL2,L_train_1,
         y_train, L_test_1,regression=True, 
         n_folds=k,mode='oof_pred',random_state=7, 
         verbose=2)


# #### layer 3
# 

# In[ ]:


#our estimator (hyper params have been found by randomized search)
ENet3=make_pipeline(RobustScaler(), ElasticNet(alpha=0.006, l1_ratio=0.0008,
                                                random_state=7))


# In[ ]:


#layer 3
L_train_3, L_test_3=stacking([ENet3],L_train_2,
         y_train, L_test_2,regression=True, 
         n_folds=k,mode='oof_pred',random_state=7, 
         verbose=1)

print(rmsle(y_train,L_train_3))


# In[ ]:


stack_pred=np.expm1(L_test_3).reshape(len(L_test_3),)

#traing predictions are in logged form 
#because the y_train is still in this form too
stack_train=L_train_3.reshape(len(L_train_3),)


# ## Weighted average ensemble
# 
# 

# In[ ]:



model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
#########################################################################
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


# **XGBoost:**

# In[ ]:


model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print(rmsle(y_train, xgb_train_pred))


# **LightGBM:**

# In[ ]:


model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))
print(rmsle(y_train, lgb_train_pred))


# ### training error

# In[ ]:


'''RMSE on the entire Train data when averaging'''

print('RMSLE score on train data:')
print(rmsle(y_train,stack_train*0.7 +xgb_train_pred*0.12+ lgb_train_pred*0.18  ))


# ### Ensemble prediction

# In[ ]:


stack_pred=stack_pred.reshape(1459,)
ensemble =stack_pred*0.7 +xgb_pred*0.12 + lgb_pred*0.18  


# In[ ]:


ensemble.shape


# ## Submission

# In[ ]:


sub = pd.DataFrame()
sub['Id'] = range(1461,1461+1459)
sub['SalePrice'] = ensemble
sub.to_csv('submission.csv',index=False)
sub.head()


# Score on the leader board :**0.11433**

# #### Please let me know if you had ideas for improving this notebook,also if have problems understanding the code ask in the comments and I will answer .Thanks for reading this note book hope it helps you !
