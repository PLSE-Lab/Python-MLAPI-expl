#!/usr/bin/env python
# coding: utf-8

# # My Journey to Third Place

# ## Submissions I made
# 
# From My Submissions page:
# 
# Submission and Description | Private Score | Public Score | Rank
# --- | --- | --- | ---
# 3rd variation for challenge (ver 1/1) | 0.46659 | 0.46800 | late entry; 3
# Russ' challenge submission (ver 25/25) | 0.47684 | 0.47698 | 3
# Russ' challenge submission (ver 24/25) | 0.47529 | 0.47565 | 3
# Russ' challenge submission (ver 22/25) | 0.48232 | 0.48313 | knocked down to 3
# Russ' challenge submission (ver 21/25) | 0.48242 | 0.48354 | 2
# Russ' challenge submission (ver 17/25) | 0.50623 | 0.50568 | 3
# Russ' challenge submission (ver 16/25) | 0.51741 | 0.52067 | 3
# Russ' challenge submission (ver 15/25) | 0.51956 | 0.51951 | 3
# Russ' challenge submission (ver 12/25) | 0.52041 | 0.52253 | 3
# 2nd shot at submission by Russ (ver 5/5) | 0.50958 | 0.51098 | 3
# Tutorial 3 in Python: Cross Validation (ver 4/5) | 0.50847 |0.51158 | 3
# Tutorial 3 in Python: Cross Validation (ver 3/5) | 0.50728 |0.50938 | 3
# Tutorial 3 in Python: Cross Validation (ver 2/5) | 0.57098 |0.57256 | 3
# Russ' challenge submission (ver 10/25) | 0.50790 | 0.50644 | 3
# Russ' challenge submission (ver 9/25) | 0.50623 | 0.50568 | 3
# Russ' challenge submission (ver 8/25) | 0.51235 | 0.51428 | 3
# Russ' challenge submission (ver 7/25) | 0.55351 | 0.55568 | 5
# Russ' challenge submission (ver 6/25) | 0.53596 | 0.53794 | 5
# Russ' challenge submission (ver 5/25) | 0.56989 | 0.57197 | 6
# Russ' challenge submission (ver 4/25) | 0.54910 | 0.55049 | 6
# 

# ## Tactics I used
# 
# * Stand on the shoulders of giants
#     - Start with someone else's kernel
#     - Run it
#     - Understand what they did
# * Where are the knobs?
#     - What might make for a better prediction
# * What's does this do?
#     - Modify a parameter or two
#     - Run it
#     - See if things are better
# 
# ## Where the weeds are
# 
# * I have no idea what this does!
# * Analysis paralysis
# * How come nobody told me before?
# * Are we there yet?
# * Follow the Golden Path (tm) of the scientific method
# * Time management
# 
# ## What worked
# 
# * Reading up on XGBoost parameter tuning
#      - [Hyperparameter tuning in XGBoost](https://cambridgespark.com/content/tutorials/hyperparameter-tuning-in-xgboost/index.html)
#      - [Getting Started with XGBoost](https://cambridgespark.com/content/tutorials/getting-started-with-xgboost/index.html)
#      - [Complete Guide to Parameter Tuning in XGBoost (with codes in Python)](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)
# * Feature selection
# * max_depth

# ## First Submission
# 
# ### Model setup
# 
# I used these as-is from [Tutorial 3 in Python: Cross Validation](https://www.kaggle.com/mmotoki/tutorial-3-in-python-cross-validation):
# * features selected = ['stock_id', 'customer_id', 'unit_price', 'invoice_id', 'integer_time']
# * max_depth = 99
# * eta = learning_rate = 0.2
# 
# ```python
# final_model = xgb.XGBRegressor(
#     max_depth = best_depth,
#     learning_rate = 0.2,
#     n_estimators = best_iter,
#     n_jobs = 4
# )
# ```
# 
# In later submissions, I removed transactions with stock_id = 3614 which is "adjust bad debt". It's only in the training data set so that should be OK. Another factor in deciding to leave them out is the large unit price values of the transactions.

# ## Best submission
# 
# ### Feature selection
# 
# I added the month as an engineered feature.
# 
# ```python
# #month of year (1, 12)
# def extract_month(date):
#     return np.array([x.split('/')[0] for x in date], dtype=int)
# train['month'] = extract_month(train['date'])
# test['month'] = extract_month(test['date'])
# ```
# 
# I played around with feature selection by trying different orderings of features. Order seemed to matter. From a previous run, parameter tuning yielded:
# ```
# Best Depth: 13 at 0.49506180000000005
# Best Iter: 199
# Best Features: ['invoice_id', 'integer_time', 'month']
# ```
# 
# ### Model setup
# 
# To save time during 'Commit and Run', I commented out the parameter tuning steps and just plugged in what I thought were the best parameters.
# 
# ```
# # for submission, skip feature and max depth selections
# best_iter = 300 # if 249 is good, 300 must be better
# best_depth = 13
# 
# final_model = xgb.XGBRegressor(
#     max_depth = best_depth,
#     learning_rate = 0.2,
#     n_estimators = best_iter,
#     n_jobs = 4
# )
# ```
# 
# Also, I was curious as to what parameters were set in my model so I printed them out.
# ```
# print(final_model)
# ```
# It said:
# ```
# XGBRegressor(
#     base_score=0.5,
#     booster='gbtree',
#     colsample_bylevel=1,
#     colsample_bytree=1, 
#     gamma=0,
#     learning_rate=0.2,
#     max_delta_step=0,
#     max_depth=13,
#     min_child_weight=1,
#     missing=None,
#     n_estimators=300,
#     n_jobs=4,
#     nthread=None,
#     objective='reg:linear',
#     random_state=0,
#     reg_alpha=0,
#     reg_lambda=1,
#     scale_pos_weight=1,
#     seed=None,
#     silent=True,
#     subsample=1
# )
# ```
# 

# ## Postscript: Using the system to learn more
# 
# I asked a challenge organizer if it would be OK to make a submission after the deadline.  I wanted to test a different approach to feature selection and parameter tuning. The late submission was my best but I still wouldn't have placed higher. Of course, that's not the point. I did learn a bit more about XGBoost.
# 
# ### Feature importance
# 
# After training an XGBoostRegressor with a training set, the model can tell us how the features rank in order of importance.
# 
# ```
# regr = xgb.sklearn.XGBRegressor(n_jobs=-1, random_state=42)
# regr.fit(X_train, y_train)
# ```
# 
# A plot is displayed using:
# 
# ```
# plt.figure(figsize=(20,15))
# xgb.plot_importance(regr, ax=plt.gca())
# ```
# 
# The features in descending order of importance are:
# * unit_price
# * customer_id
# * invoice_id
# * integer_time
# * stock_id
# * month
# 

# ### Standing on the shoulders of giants
# 
# I followed the steps in [Hyperparameter tuning in XGBoost](https://cambridgespark.com/content/tutorials/hyperparameter-tuning-in-xgboost/index.html).
# 

# #### Start with a baseline model
# ```python
# # parameters we're going to tune
# params = {
#     'max_depth': 6,
#     'min_chile_weight': 1,
#     'eta': 3,
#     'subsample': 1,
#     'colsample_bytree': 1,
#     'objective': 'reg:linear',
# }
# ```

# #### Tune num_boost_round and early_stopping_rounds
# ```python
# params['eval_metric'] = 'rmse'
# num_boost_round = 999
# 
# model = xgb.train(
#     params,
#     dtrain,
#     num_boost_round = num_boost_round,
#     evals = [(dtest, "Test")],
#     early_stopping_rounds = 10
# )
# ```

# #### Tune max_depth and min_child_weight
# ```python
# gridsearch_params = [
#     (max_depth, min_child_weight)
#     for max_depth in range(9,12)
#     for min_child_weight in range(5,8)
# ]
# ```
# 
# Output:
# ```
# ...
# CV with max_depth=11, min_child_weight=7
# 	RMSE 2.8512514 for 0 rounds
# Best params: 9, 5, RMSE: 2.8359496
# ```

# #### Tune subsample and colsample_bytree
# ```python
# gridsearch_params = [
#     (subsample, colsample)
#     for subsample in [i/10. for i in range(7,11)]
#     for colsample in [i/10. for i in range(7,11)]
# ]
# ```
# 
# Output:
# ```
# ...
# CV with subsample=0.7, colsample=0.7
# 	RMSE 2.8031129999999997 for 0 rounds
# Best params: 0.9, 0.8, RMSE: 2.800571
# ```

# #### Tune eta (learning rate)
# 
# ```python
# min_rmse = float('Inf')
# best_params = None
# 
# def cross_validation(params, dtrain, num_boost_round):
#     return xgb.cv(
#         params,
#         dtrain,
#         num_boost_round = num_boost_round,
#         seed=42,
#         nfold=5,
#         metrics={'rmse'},
#         early_stopping_rounds=10
#     )
# 
# #for eta in [.3, .2, .1, .05, .01, .005]:
# for eta in [.3, .2, .1]:
#     print('CV with eta={}'.format(eta))
#     
#     params['eta'] = eta
# 
#     %time cv_results = cross_validation(params, dtrain, num_boost_round)
# 
#     mean_rmse = cv_results['test-rmse-mean'].min()
#     boost_rounds = cv_results['test-rmse-mean'].idxmin()
#     print("\tRMSE {} for {} rounds".format(mean_rmse, boost_rounds))
#     if mean_rmse < min_rmse:
#         min_rmse = mean_rmse
#         best_params = eta
#         
# print('Best params: {}, RMSE: {}'.format(best_params, min_rmse))
# ```
# 
# Output:
# ```
# CV with eta=0.3
# CPU times: user 57min 47s, sys: 17.8 s, total: 58min 5s
# Wall time: 14min 33s
# 	RMSE 0.482008 for 877 rounds
# CV with eta=0.2
# CPU times: user 53min 30s, sys: 16 s, total: 53min 46s
# Wall time: 13min 28s
# 	RMSE 0.4765396 for 998 rounds
# CV with eta=0.1
# CPU times: user 51min 7s, sys: 15.1 s, total: 51min 22s
# Wall time: 12min 52s
# 	RMSE 0.48975539999999995 for 998 rounds
# Best params: 0.2, RMSE: 0.4765396
# ```

# #### Train our model
# 
# ```python
# Parameters: {
#     'max_depth': 9,
#     'min_chile_weight': 1,
#     'eta': 0.2,
#     'subsample': 0.9,
#     'colsample_bytree': 0.8,
#     'objective': 'reg:linear',
#     'eval_metric': 'rmse',
#     'min_child_weight': 5
# }
# 
# 
# final_model = xgb.train(
#     params,
#     dtrain,
#     num_boost_round = num_boost_round,
#     evals = [(dtest, "Test")],
#     early_stopping_rounds=10
# )
# ```
# 
# Output:
# ```
# Best RMSE: 0.47 in 999 rounds
# ```
# 
# 

# #### Make our predictions
# 
# ```python
# dtest = xgb.DMatrix(test[predictors], label=y)
# test_preds = np.expm1(final_model.predict(dtest))
# ```

# In[ ]:




