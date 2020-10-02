#!/usr/bin/env python
# coding: utf-8

# ## Supervised Regression- House Price Data
# 
# _By Nick Brooks_
# 

# In[ ]:


# General
import numpy as np
import pandas as pd
import os
import scipy.stats as st
import pickle
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Evalaluation
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer

# Grid
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

# Models
# Linear Regression
from sklearn import linear_model
from sklearn.linear_model import Ridge

# XGBoost
import xgboost as xgb
from xgboost.sklearn import XGBRegressor  


# In[ ]:


import os
print(os.listdir('../input/feature-engineering-and-pre-processing-house-data'))


# In[ ]:


# Train
train_df = pd.read_csv("../input/feature-engineering-and-pre-processing-house-data/house_train.csv", index_col='Id')
traindex = train_df.index
# Log conversion
y = np.log(train_df['SalePrice'])
train_df.drop("SalePrice",axis=1,inplace=True)

# Test
test_df = pd.read_csv("../input/feature-engineering-and-pre-processing-house-data/house_test.csv", index_col='Id')
testdex = test_df.index

# Combine
df = pd.concat([train_df,test_df],axis=0)

print("Train Shape:", train_df.shape)
print("Test Shape:",test_df.shape)
print("ALL Shape:", df.shape)
del test_df, train_df


# In[ ]:


from sklearn.preprocessing import LabelEncoder
# Encoder:
encode = df.loc[:,df.dtypes=="object"].columns
lbl = LabelEncoder()
for col in encode:
     df[col] = lbl.fit_transform(df[col].astype(str))
# I might want to add a scaler..


# In[ ]:


submission_df = pd.DataFrame(index=testdex)

# Indepedent and Dependent
X = df.loc[traindex,:]
testing = df.loc[testdex,:]
del df

# use train/test split with different random_state values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("X_train Shape: {}".format(X_train.shape), "\ny_train Shape: {}".format(y_train.shape),
      "\nX_test Shape: {}".format(X_test.shape), "\ny_test Shape: {}".format(y_test.shape))

print("\nDo Train and Submission Set Columns Match?\n{}".format(X.columns.equals(testing.columns)))


# In[ ]:


# Hyper-Parameter
n_inter = 25
cv = 5
rstate = 23
score_name = "Root Mean Square Error"


# In[ ]:


# Define a function to calculate Root Mean Sqaure Error
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true-y_pred)**2))

# Define a function to calculate negative RMSE (as a score)
def nrmse(y_true, y_pred):
    return -1.0*rmse(y_true, y_pred)

#neg_rmse = make_scorer(nrmse)
scoring = make_scorer(rmse, greater_is_better=False)


# In[ ]:


## Helpers


# In[ ]:


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

# Storage for Model and Results
results = pd.DataFrame(columns=['Model','Para','Test_Score','CV Mean','CV STDEV'])
def save(model, modelname):
    global results
    model.best_estimator_.fit(X, y)
    submission =  np.exp(model.predict(testing))
    
    df = pd.DataFrame({'Id':testing.index, 
                        'SalePrice':submission})
    df.to_csv("{}.csv".format(modelname),header=True,index=False)
    submission_df[modelname] = submission
    
    model.best_estimator_.fit(X_train, y_train)
    top = np.flatnonzero(grid.cv_results_['rank_test_score'] == 1)
    CV_scores = grid.cv_results_['mean_test_score'][top]*-1
    STDev = grid.cv_results_['std_test_score'][top]
    Test_scores = rmse(y_test, model.predict(X_test))
    
    # CV and Save Scores
    results = results.append({'Model': modelname,'Para': model.best_params_,'Test_Score': Test_scores,
                             'CV Mean':CV_scores, 'CV STDEV': STDev}, ignore_index=True)
    
    # Print Evaluation
    print("\nEvaluation Method: {}".format(score_name))
    print("Optimal Model Parameters: {}".format(grid.best_params_))
    print("Training RMSE: ", rmse(y_train, model.predict(X_train)))
    print("Train CV Accuracy: %0.2f (+/- %0.2f) [%s]" % (CV_scores, STDev, modelname))
    print('Test_Score:', Test_scores)


# # Models
# ## Linear Regression

# In[ ]:


linear_model.LinearRegression().get_params().keys()


# In[ ]:


model = linear_model.LinearRegression()
score = cross_val_score(model, X, y, cv=5, scoring=scoring)
print("CV Mean: {:.3f} RMSE +/- {:.5f}".format(abs(score.mean()), score.std()))


# ## Ridge Regression

# In[ ]:


Ridge().get_params().keys()


# In[ ]:


model = Ridge()

alpha= st.beta(10, 1)
alpha = [1000,100,10, 1, 0.1, 0.01, 0.001,0.0001]
alpha = np.logspace(4,-4,10)

param_grid = {'alpha': alpha,
             'fit_intercept': [True,False],
              'normalize':[True,False],
              #'solver':,
              #'tol': 
             }

grid = RandomizedSearchCV(model, param_grid,
                          cv=cv, verbose=1, scoring=scoring,
                         n_iter=30, random_state=rstate)

grid.fit(X_train, y_train)
save(grid, "Ridge")


# In[ ]:


report(grid.cv_results_)


# ## XGBoost

# In[ ]:


# Human Analog Model
# https://www.kaggle.com/humananalog/xgboost-lasso/code
regr = xgb.XGBRegressor()

score = cross_val_score(regr, X, y, cv=5, scoring=scoring)
print("CV Mean: {:.3f} RMSE +/- {:.5f}".format(abs(score.mean()), score.std()))


# In[ ]:


regr.fit(X_train, y_train)

f, ax = plt.subplots(figsize=[8,12])
xgb.plot_importance(regr,max_num_features=50,ax=ax)
plt.show


# In[ ]:


# Run prediction on training set to get a rough idea of how well it does.
y_pred = regr.predict(X_train)
print("XGBoost score on training set: ", rmse(y_train, y_pred))
print("XGBoost score on training set: ", rmse(y_test, regr.predict(X_test)))

submission_df["XGBOOST"] = np.exp(regr.predict(testing))


# In[ ]:


regr.evals_result


# ## Light GBM

# In[ ]:


import lightgbm as lgb
lgbm_params = {
    'boosting_type': 'gbdt',
    'objective': "regression",
    'metric': 'rmse',
    'max_bin': 300,
    'max_depth': 5,
    'num_leaves': 200,
    'learning_rate': 0.01,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 10,
    'verbose': 0,
    'num_threads': 1,
    'lambda_l2': 3,
    'min_gain_to_split': 0,
}

import time
modelstart= time.time()

predictors = list(X_train.columns)
categorical = list(encode)
dtrain = lgb.Dataset(X_train.values, label=y_train.values,
                    feature_name = predictors,
                    categorical_feature=categorical
                    )
dvalid = lgb.Dataset(X_test.values, label=y_test.values,
                      feature_name=predictors,
                      categorical_feature=categorical
                      )
lgb_reg = lgb.train(
    lgbm_params,
    dtrain,
    num_boost_round=4000,
    valid_sets = [dtrain,dvalid],
    valid_names= ["train","valid"],
    early_stopping_rounds=75,
    verbose_eval = 75)


# In[ ]:


f, ax = plt.subplots(figsize=[7,10])
lgb.plot_importance(lgb_reg, max_num_features=50, ax=ax)
plt.title("Light GBM Feature Importance")
plt.show()


# In[ ]:


lgbmpred = np.exp(lgb_reg.predict(testing))
submission_df["lgbmpred"] = lgbmpred
lgbm_sub = pd.DataFrame(lgbmpred,columns=["SalePrice"],index=testdex)
lgbm_sub.to_csv("lgbm_sub.csv",index=True)
print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))


# ## Regularized Linear Models
# 
# This is somebody else's work

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 6


alpha = 0.1
lasso = Lasso(alpha=alpha)

score = cross_val_score(lasso, X, y, cv=5, scoring=scoring)
print("Lasso Net CV Mean: {:.3f} RMSE +/- {:.5f}".format(abs(score.mean()), score.std()))

y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
submission_df["lasso"] = np.exp(lasso.predict(testing))
r2_score_lasso = r2_score(y_test, y_pred_lasso)
print(lasso)
print("r^2 on test data : %f" % r2_score_lasso)

# #############################################################################
# ElasticNet
from sklearn.linear_model import ElasticNet

enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
y_pred_enet = enet.fit(X_train, y_train).predict(X_test)

score = cross_val_score(enet, X, y, cv=5, scoring=scoring)
print("Elastic Net CV Mean: {:.3f} RMSE +/- {:.5f}".format(abs(score.mean()), score.std()))

submission_df["lasso"] = np.exp(enet.predict(testing))
r2_score_enet = r2_score(y_test, y_pred_enet)
print(enet)
print("r^2 on test data : %f" % r2_score_enet)

plt.plot(enet.coef_, color='lightgreen', linewidth=2,
         label='Elastic net coefficients')
plt.plot(lasso.coef_, color='gold', linewidth=2,
         label='Lasso coefficients')
#plt.plot(coef, '--', color='navy', label='original coefficients')
plt.legend(loc='best')
plt.title("Lasso R^2: %f, Elastic Net R^2: %f"
          % (r2_score_lasso, r2_score_enet))
plt.show()


# ## Blend

# In[ ]:


submission_df.head()


# In[ ]:


submission_df.head()


# In[ ]:


out = pd.DataFrame(submission_df[["lgbmpred","XGBOOST"]].mean(axis=1)).rename(columns={0:"SalePrice"})
out.to_csv("averaged_sub.csv", index=True,header=True)
out.head()

