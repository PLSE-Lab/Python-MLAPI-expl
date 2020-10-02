#!/usr/bin/env python
# coding: utf-8

# # Light GBM with engineered features
# 
# ** Note to self**: this kernel will purposfully avoid using the feature visitStartTime directly as this appears to lead to overfit models on the public Leader Board. i.e. very good results for no apparent reason... 

# ### Libraries 

# In[ ]:


import os
import numpy as np
import pandas as pd
import time
import warnings
import datetime

#Plotting
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

#Sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

#lgm and graph viz
import graphviz 
import lightgbm as lgb

warnings.filterwarnings('ignore')


# In[ ]:


os.listdir('../input/kernel-for-saving-files')


# ## Loading files which were partially preprocessed from previous kernel
# 
# - Categories are label encoded
# - Local time field is calculated (_local_hourofday)
# - Time since last visit is already calculated

# In[ ]:


get_ipython().run_cell_magic('time', '', "train_df = pd.read_pickle('../input/kernel-for-saving-files/train_flat_local_cat_enc.pkl')\ntest_df = pd.read_pickle('../input/kernel-for-saving-files/test_flat_local_cat_enc.pkl')")


# In[ ]:


train_df.info()


# # Feature engineering 
# 
# ## Columns definitions

# In[ ]:


# Extract target values and Ids
cat_cols = ['channelGrouping','device.browser',
       'device.deviceCategory', 'device.isMobile', 'device.operatingSystem',
       'geoNetwork.city', 'geoNetwork.continent', 'geoNetwork.country',
       'geoNetwork.metro', 'geoNetwork.networkDomain', 'geoNetwork.region',
       'geoNetwork.subContinent','trafficSource.adContent',
       'trafficSource.adwordsClickInfo.adNetworkType',
       'trafficSource.adwordsClickInfo.gclId',
       'trafficSource.adwordsClickInfo.isVideoAd',
       'trafficSource.adwordsClickInfo.page',
       'trafficSource.adwordsClickInfo.slot', 'trafficSource.campaign',
       'trafficSource.isTrueDirect', 'trafficSource.keyword',
       'trafficSource.medium', 'trafficSource.referralPath',
       'trafficSource.source'  ]


num_cols = ['visitNumber', 'totals.bounces', 'totals.hits',
            'totals.newVisits', 'totals.pageviews' ]

interaction_cols = ['totals.hits / totals.pageviews', 'totals.hits * totals.pageviews','visitNumber / totals.hits']

visitStartTime = ['visitStartTime']

time_cols = ['_dayofweek', '_monthofyear', '_dayofyear', '_local_hourofday', '_time_since_last_visit']

ID_cols = ['date', 'fullVisitorId', 'sessionId', 'visitId']

target_col = ['totals.transactionRevenue']


# ## Time features
# 

# ### Other simple date based features 

# In[ ]:


train_df['_dayofweek'] = train_df['visitStartTime'].dt.dayofweek
train_df['_monthofyear'] = train_df['visitStartTime'].dt.month
train_df['_dayofyear'] = train_df['visitStartTime'].dt.dayofyear
#train_df['_dayofmonth'] = train_df['visitStartTime'].dt.day

test_df['_dayofweek'] = test_df['visitStartTime'].dt.dayofweek
test_df['_monthofyear'] = test_df['visitStartTime'].dt.month
test_df['_dayofyear'] = test_df['visitStartTime'].dt.dayofyear
#test_df['_dayofmonth'] = test_df['visitStartTime'].dt.day


# ## Interaction feature
# 
# Only a few are selected currently..

# In[ ]:


get_ipython().run_cell_magic('time', '', "from itertools import combinations\n\nto_interact_cols = ['visitNumber', 'totals.hits', 'totals.pageviews']\n\n#Numeric as float\nfor n in [num_cols + time_cols]:\n    train_df[n] = train_df[n].fillna(0).astype('float')\n    test_df[n] = test_df[n].fillna(0).astype('float')\n    \n\n\ndef numeric_interaction_terms(df, columns):\n    for c in combinations(columns,2):\n        df['{} / {}'.format(c[0], c[1]) ] = df[c[0]] / df[c[1]]\n        df['{} * {}'.format(c[0], c[1]) ] = df[c[0]] * df[c[1]]\n        df['{} - {}'.format(c[0], c[1]) ] = df[c[0]] - df[c[1]]\n    return df\n\n\ntrain_df = numeric_interaction_terms(train_df,to_interact_cols )\ntest_df = numeric_interaction_terms(test_df,to_interact_cols )")


# In[ ]:


train_df.head()


# # Pre-processing

# ## Label encoding
# Already done

# ## Target, index and extraction of datasets

# In[ ]:


train_df['totals.transactionRevenue'] = train_df['totals.transactionRevenue'].fillna(0).astype('float')

#Index
train_idx = train_df['fullVisitorId']
test_idx = test_df['fullVisitorId']

#Targets
train_target = np.log1p(train_df.groupby("fullVisitorId")["totals.transactionRevenue"].sum())
train_y = np.log1p(train_df["totals.transactionRevenue"].values)

#Datasets
train_X = train_df[cat_cols + num_cols + time_cols + interaction_cols].copy()
test_X = test_df[cat_cols + num_cols + time_cols + interaction_cols].copy()

print(train_X.shape)
print(test_X.shape)


# In[ ]:


train_X.info()


# # Light GBM 
# ## Initialize (Sklearn wrapper)

# In[ ]:


from lightgbm import LGBMRegressor

#Initialize LGBM
gbm = LGBMRegressor(objective = 'regression', 
                     boosting_type = 'dart', 
                     metric = 'rmse',
                     n_estimators = 10000, #10000
                     num_leaves = 54, #10
                     learning_rate = 0.005, #0.01
                     #bagging_fraction = 0.9,
                     #feature_fraction = 0.3,
                     bagging_seed = 0,
                     max_depth = 10,
                     reg_alpha = 0.436193,
                     reg_lambda = 0.479169,
                     colsample_bytree = 0.508716,
                     min_split_gain = 0.024766
                    )


# ## Fit the model
# **In a nutshell**: K-fold training where each fold is used once for early stopping validation. At each fold, a test prediction is made using the trained model. Final prediction is an average of the K predictions
# 
# **Steps:**
# - Fit the LGBM model K times on the dataset - the Kth fold
#  - For each fitted model, predict on validation set (oof_pred) and on test set (sub_preds)
#  - Average presub_preds for final predictions
# 
# Idea for averaged models comes from: https://www.kaggle.com/sz8416/lb-1-4439-gacr-prediction-eda-lgb-baseline

# In[ ]:


get_ipython().run_cell_magic('time', '', "#Initilization\nall_K_fold_results = []\nkf = KFold(n_splits=5, shuffle = True)\noof_preds = np.zeros(train_X.shape[0])\nsub_preds = np.zeros(test_X.shape[0])\n\n\nfor dev_index, val_index in kf.split(train_X):\n    X_dev, X_val = train_X.iloc[dev_index], train_X.iloc[val_index]\n    y_dev, y_val = train_y[dev_index], train_y[val_index]\n\n    #Fit the model\n    model = gbm.fit(X_dev,y_dev, eval_set=[(X_val, y_val)],verbose = 100, \n                    eval_metric = 'rmse', early_stopping_rounds = 100) #100\n    \n    #Predict out of fold \n    oof_preds[val_index] = gbm.predict(X_val, num_iteration= model.best_iteration_)\n    \n    oof_preds[oof_preds < 0] = 0\n    \n    #Predict on test set based on current fold model. Average results\n    sub_prediction = gbm.predict(test_X, num_iteration= model.best_iteration_) / kf.n_splits\n    sub_prediction[sub_prediction<0] = 0\n    sub_preds = sub_preds + sub_prediction\n    \n    #Save current fold values\n    fold_results = {'best_iteration_' : model.best_iteration_, \n                   'best_score_' : model.best_score_['valid_0']['rmse'], \n                   'evals_result_': model.evals_result_['valid_0']['rmse'],\n                   'feature_importances_' : model.feature_importances_}\n\n    all_K_fold_results.append(fold_results.copy())\n    \n\nresults = pd.DataFrame(all_K_fold_results)\n\n")


# ## Visualization, RMSE and saving utility functions 
# A helper function which plots the RMSE as a function of iterations, a box plot of the RMSE, and the average feature importance (with std error bars)
# 
# **Note**: exponentiating the predictions made on $log(y)$ might not be mathematically valid... to be investigated further. See this discussion https://www.kaggle.com/c/ga-customer-revenue-prediction/discussion/67206

# In[ ]:


def RMSE_log_sum(pred_val, val_df):
    #set negative values to zero
    pred_val[pred_val < 0] = 0
    
    #Build new dataframe
    val_pred_df = pd.DataFrame(data = {'fullVisitorId': val_df['fullVisitorId'].values, 
                                       'transactionRevenue': val_df['totals.transactionRevenue'].values,
                                      'predictedRevenue':np.expm1(pred_val) })
    #Compute sum
    val_pred_df = val_pred_df.groupby('fullVisitorId').sum().reset_index()

    mse_log_sum = mean_squared_error( np.log1p(val_pred_df['transactionRevenue'].values), 
                             np.log1p(val_pred_df['predictedRevenue'].values)  )

    #print('log (sum + 1): ',np.sqrt(mse_log_sum))
    return np.sqrt(mse_log_sum)


def save_submission(pred_test, test_df, file_name):
    #Zero negative predictions
    pred_test[pred_test < 0] = 0
    
    #Create temporary dataframe
    sub_df = pd.DataFrame(data = {'fullVisitorId':test_df['fullVisitorId'], 
                             'predictedRevenue':np.expm1(pred_test)})
    sub_df = sub_df.groupby('fullVisitorId').sum().reset_index()
    sub_df.columns = ['fullVisitorId', 'predictedLogRevenue']
    sub_df['predictedLogRevenue'] = np.log1p(sub_df['predictedLogRevenue'])
    sub_df.to_csv(file_name, index = False)

    
def visualize_results(results):
#Utility function to plot fold loss and best model feature importance
    plt.figure(figsize=(16, 12))

    #----------------------------------------
    # Plot validation loss
    plt.subplot(2,2,1)

    for K in range(results.shape[0]):
        plt.plot(np.arange(len(results.evals_result_[K])), results.evals_result_[K], label = 'fold {}'.format(K))

    plt.xlabel('Boosting iterations')
    plt.ylabel('RMSE')
    plt.title('Validation loss vs boosting iterations')
    plt.legend()

    #----------------------------------------
    # Plot box plot of RMSE
    plt.subplot(2, 2, 2)    
    scores = results.best_score_
    plt.boxplot(scores)
    rmse_mean = np.mean(scores)
    rmse_std = np.std(scores)
    plt.title('RMSE Mean:{:.3f} Std: {:.4f}'.format(rmse_mean,rmse_std ))
    
    #----------------------------------------
    # Plot feature importance
    #feature_importance = results.sort_values('best_score_').feature_importances_[0]
    df_feature_importance = pd.DataFrame.from_records(results.feature_importances_)
    feature_importance = df_feature_importance.mean()
    std_feature_importance = df_feature_importance.std()
    
    # make importances relative to max importance
    #feature_importance = 100.0 * (mean_feature_importance / mean_feature_importance.sum())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(2, 1, 2)
    plt.bar(pos, feature_importance[sorted_idx], align='center', yerr = std_feature_importance)
    xlabels = [ train_X.columns.values[i] for i in sorted_idx]
    plt.xticks(pos, xlabels, rotation = 90)
    plt.xlabel('Feature')
    plt.ylabel('Avg Importance score')
    plt.title('Mean Feature Importance over K folds') 
    
    plt.show()


# In[ ]:


print('Session level CV score: ', np.mean(results.best_score_))
print('User level CV score: ', RMSE_log_sum(oof_preds, train_df))


# In[ ]:


results.evals_result_


# ## Validation RMSE and feature importance 

# In[ ]:


visualize_results(results)


# ## Visualize the first decision tree
# 

# In[ ]:


import graphviz 
dot_data = lgb.create_tree_digraph(model, tree_index = 1,show_info=['split_gain'])

graph = graphviz.Source(dot_data)  
graph 


# # Error analysis
# ## Distributions of true and predicted log revenues 

# In[ ]:


error_df = pd.DataFrame(data = {'visitStartTime':train_df['visitStartTime'],'fullVisitorId':train_df['sessionId'], 
                                'True_log_revenue' : np.log1p(train_df['totals.transactionRevenue']), 
                                'Predicted_log_revenue':oof_preds  })

error_df['Difference'] = error_df['True_log_revenue'] - error_df['Predicted_log_revenue']
error_df['True_is_non_zero'] = error_df['True_log_revenue'] > 0
#temp_df.columns = ['fullVisitorId', 'predictedLogRevenue']
#sub_df['predictedLogRevenue'] = np.log1p(sub_df['predictedLogRevenue'])
#sub_df.to_csv(file_name, index = False)
error_df.head(100).sort_values('True_log_revenue')


# In[ ]:


fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize = (20,7))


sns.distplot(error_df[error_df['True_is_non_zero'] == False]['True_log_revenue'], ax = ax1, label = 'true')
sns.distplot(error_df[error_df['True_is_non_zero'] == False ]['Predicted_log_revenue'], ax = ax1, label = 'pred')
ax1.legend()
ax1.set_ylim(0,.1)
ax1.set_xlabel('Log revenue (session)')
ax1.set_title('Distribution of log revenues for sessions with zero true revenue ')

sns.distplot(error_df[error_df['True_is_non_zero'] == True]['True_log_revenue'], ax = ax2, label = 'true')
sns.distplot(error_df[error_df['True_is_non_zero'] == True ]['Predicted_log_revenue'], ax = ax2, label = 'pred')
ax2.legend()
ax2.set_ylim(0,.5)
ax2.set_xlabel('Log revenue (session)')
ax2.set_title('Distribution of log revenues for sessions with non zero true revenue ')

plt.show()


# In[ ]:


sorted_non_zero = error_df[error_df['True_is_non_zero'] == True].sort_values('visitStartTime')


plt.figure(figsize = (20,15))
plt.subplot(2,2,1)
plt.plot(sorted_non_zero.visitStartTime, sorted_non_zero.True_log_revenue , label = 'True')
plt.plot(sorted_non_zero.visitStartTime, sorted_non_zero.Predicted_log_revenue , alpha = .5, label = 'Pred')
plt.title('Log revenue over time (non zero true sessions only)')
plt.legend()
plt.xlabel('Time: sessions')

plt.subplot(2,2,2)
daily_error_non_zero_df = sorted_non_zero.set_index('visitStartTime', drop = True).resample('D').mean()
plt.plot(daily_error_non_zero_df.index, daily_error_non_zero_df.True_log_revenue , label = 'True')
plt.plot(daily_error_non_zero_df.index, daily_error_non_zero_df.Predicted_log_revenue , label = 'Pred')
plt.title('Daily average log revenue (non zero true sessions only)')

plt.subplot(2,2,3)
weekly_error_df = error_df.set_index('visitStartTime', drop = True).resample('W').mean()
plt.plot(weekly_error_df.index, weekly_error_df.True_log_revenue , label = 'True')
plt.plot(weekly_error_df.index, weekly_error_df.Predicted_log_revenue , label = 'Pred')
plt.title('Weekly average log revenue (all session)')


plt.subplot(2,2,4)
daily_error_df = error_df.set_index('visitStartTime', drop = True).resample('D').mean()
plt.plot(daily_error_df.index, daily_error_df.True_log_revenue , label = 'True')
plt.plot(daily_error_df.index, daily_error_df.Predicted_log_revenue , label = 'Pred')
plt.title('Daily average log revenue (all session)')

plt.legend()
plt.show()


# In[ ]:


sorted_non_zero = error_df[error_df['True_is_non_zero'] == True].sort_values('visitStartTime')
sorted_zero = error_df[error_df['True_is_non_zero'] == False].sort_values('visitStartTime')


plt.figure(figsize = (20,5))
plt.subplot(1,3,1)
ts_error_df = error_df.set_index('visitStartTime', drop = True)
difference_rev_df = error_df.sort_values('visitStartTime')
plt.plot(error_df.visitStartTime, error_df.Difference , label = 'True - predicted', color = 'grey')
plt.title('Train - Pred (log rev) for all sessions')

plt.subplot(1,3,2)
plt.plot(sorted_non_zero.visitStartTime, sorted_non_zero.Difference , label = 'True - predicted',
         color = 'grey')
plt.title('Train - Pred for non zero sessions only')

plt.subplot(1,3,3)
plt.plot(sorted_zero.visitStartTime, sorted_zero.Difference,
         color = 'grey')
plt.title('Train - Pred for zero sessions only')

plt.legend()
plt.show()


# In[ ]:



sns.jointplot(x="True_log_revenue", y="Predicted_log_revenue", data=sorted_non_zero)
display('Joint distribution of log rev for non zero sessions only')

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


pred_test[pred_test < 0] = 0
    
    #Create temporary dataframe
    sub_df = pd.DataFrame(data = {'fullVisitorId':test_df['fullVisitorId'], 
                             'predictedRevenue':np.expm1(pred_test)})
    sub_df = sub_df.groupby('fullVisitorId').sum().reset_index()
    sub_df.columns = ['fullVisitorId', 'predictedLogRevenue']
    sub_df['predictedLogRevenue'] = np.log1p(sub_df['predictedLogRevenue'])
    sub_df.to_csv(file_name, index = False)


# ## Save submission 

# In[ ]:


save_submission(sub_preds, test_df, 'submission.csv')


# In[ ]:




