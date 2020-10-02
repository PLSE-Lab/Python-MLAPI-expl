#!/usr/bin/env python
# coding: utf-8

# # Regression with Bike Sharing Demand Data

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import os
print(os.listdir("../input"))


# ## 1. Data Loading and Preprocessing

# In[ ]:


bike_df = pd.read_csv('../input/train.csv')
bike_df.shape


# In[ ]:


bike_df.info()


# In[ ]:


bike_df.head()


# The 'datetime' feature needs to be transformed as an object and to be splited into year, month, day and time. 

# In[ ]:


# Transform string into datetime type
bike_df['datetime'] = bike_df['datetime'].apply(pd.to_datetime)

# Extract year, month, day and time from the datetime type
bike_df['year'] = bike_df['datetime'].apply(lambda x : x.year)
bike_df['month'] = bike_df['datetime'].apply(lambda x : x.month)
bike_df['day'] = bike_df['datetime'].apply(lambda x : x.day)
bike_df['hour'] = bike_df['datetime'].apply(lambda x : x.hour)
bike_df.head()


# Now, I will delete the datetime feature. 
# 
# I will also delete casual and registered features as the sum of casual and registered equals to count.

# In[ ]:


bike_df.drop(['datetime', 'casual', 'registered'], axis=1, inplace=True)


# I will create a prediction performance evaluation function. 

# In[ ]:


from sklearn.metrics import mean_squared_error, mean_absolute_error

# Calculate RMSLE (Root Mean Square Log Error) with using not log() but log1p() due to issues including NaN
def rmsle(y, pred):
    log_y = np.log1p(y)
    log_pred = np.log1p(pred)
    squared_error = (log_y - log_pred) ** 2
    rmsle = np.sqrt(np.mean(squared_error))
    return rmsle

# Calculate RMSE
def rmse(y, pred):
    return np.sqrt(mean_squared_error(y, pred))

# Calculate MSE, RMSE and RMSLE
def evaluate_regr(y, pred):
    rmsle_val = rmsle(y, pred)
    rmse_val = rmse(y, pred)
    mse_val = mean_absolute_error(y, pred)
    print('RMSLE: {:.3f}, RMSE: {:.3f}, MSE: {:.3f}'.format(rmsle_val, rmse_val, mse_val))


# ## 2. Log Transformation, Feature Encoding and Model training/prediction/evaluation

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso

y_target = bike_df['count']
X_features = bike_df.drop(['count'], axis=1, inplace=False)

X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.3, random_state=2019)

lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)
pred = lr_reg.predict(X_test)

evaluate_regr(y_test, pred)


# Considering count values, such prediction errors seems to be relatively high. 
# 
# Let's check actual and predicted values in top 5 errors!

# In[ ]:


def get_top_error_data(y_test, pred, n_tops=5):
    result_df = pd.DataFrame(y_test.values, columns=['real_count'])
    result_df['predicted_count'] = np.round(pred)
    result_df['diff'] = np.abs(result_df['real_count'] - result_df['predicted_count'])
    
    print(result_df.sort_values('diff', ascending=False)[:n_tops])
    
get_top_error_data(y_test, pred, n_tops=5)


# As I mentioned, prediction errors are so high. 
# 
# Let's check Target distribution. 

# In[ ]:


y_target.hist()


# The Target distribution is skewed!
# 
# Log transformation fot the Target seems to be needed. 

# In[ ]:


y_log_transform = np.log1p(y_target)
y_log_transform.hist()


# Skewness is somewhat improved.
# 
# Let's train the model again!

# In[ ]:


y_target_log = np.log1p(y_target)

X_train, X_test, y_train, y_test = train_test_split(X_features, y_target_log, test_size=0.3, random_state=2019)

lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)
pred = lr_reg.predict(X_test)

# Convert the transformed y_test values into the original values
y_test_exp = np.expm1(y_test)

# Convert the transformed predicted values into the original values
pred_exp = np.expm1(pred)

evaluate_regr(y_test_exp, pred_exp)


# RMSLE is lower, but RMSE is higher. 
# 
# Let's check coefficient values of features!

# In[ ]:


coef = pd.Series(lr_reg.coef_, index=X_features.columns)
coef_sort = coef.sort_values(ascending=False)
sns.barplot(x=coef_sort.values, y=coef_sort.index)


# Year feature has an exceptional coefficient value, which results from being int type. 
# 
# So, categorical features including year need to be transformed into one-hot encoding. 

# In[ ]:


X_features_ohe = pd.get_dummies(X_features, columns=['year', 'month', 'hour', 'holiday', 'workingday', 'season', 'weather'])


# In[ ]:


X_features_ohe.head()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_features_ohe, y_target_log, test_size=0.3, random_state=2019)

def get_model_predict(model, X_train, X_test, y_train, y_test, is_expm1=False):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    if is_expm1:
        y_test = np.expm1(y_test)
        pred = np.expm1(pred)
    print('###', model.__class__.__name__, '###')
    evaluate_regr(y_test, pred)

lr_reg = LinearRegression()
ridge_reg = Ridge(alpha=10)
lasso_reg = Lasso(alpha=0.01)

for model in [lr_reg, ridge_reg, lasso_reg]:
    get_model_predict(model, X_train, X_test, y_train, y_test, is_expm1=True)


# After one-hot encoding, prediction performance is improved. 
# 
# Let's check features with high coefficient values.

# In[ ]:


coef = pd.Series(lr_reg.coef_, index=X_features_ohe.columns)
coef_sort = coef.sort_values(ascending=False)[:15]
sns.barplot(x=coef_sort.values, y=coef_sort.index)


# Season, month and weather features have high coefficient values.
# 
# Let's apply Regression Tree models including RandomForest, GBM, XGBoost and LightGBM.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_features_ohe, y_target_log, test_size=0.3, random_state=2019)

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

rf_reg = RandomForestRegressor(n_estimators=500)
gbm_reg = GradientBoostingRegressor(n_estimators=500)
xgb_reg = XGBRegressor(n_estimators=500)
lgbm_reg = LGBMRegressor(n_estimators=500)

for model in [rf_reg, gbm_reg, xgb_reg, lgbm_reg]:
    get_model_predict(model, X_train, X_test, y_train, y_test, is_expm1=True)


# Regression prediction performances are improved!

# ## 3. Prediction on Test Set

# In[ ]:


submission = pd.read_csv('../input/sampleSubmission.csv')


# In[ ]:


submission.shape


# In[ ]:


submission.head()


# In[ ]:


X_test = pd.read_csv('../input/test.csv')
X_test.head()


# In[ ]:


X_test.shape


# In[ ]:


# Transform string into datetime type
X_test['datetime'] = X_test['datetime'].apply(pd.to_datetime)

# Extract year, month, day and time from the datetime type
X_test['year'] = X_test['datetime'].apply(lambda x : x.year)
X_test['month'] = X_test['datetime'].apply(lambda x : x.month)
X_test['day'] = X_test['datetime'].apply(lambda x : x.day)
X_test['hour'] = X_test['datetime'].apply(lambda x : x.hour)
X_test.head()


# In[ ]:


X_test.drop(['datetime'], axis=1, inplace=True)
X_test.head()


# In[ ]:


X_test.shape


# In[ ]:


X_test_ohe = pd.get_dummies(X_test, columns=['year', 'month', 'hour', 'holiday', 'workingday', 'season', 'weather'])
X_test_ohe.head()


# In[ ]:


prediction = lgbm_reg.predict(X_test_ohe)


# In[ ]:


prediction


# In[ ]:


submission['count'] = np.round(prediction, 0).astype(int)


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('./My_submission.csv', index=False)


# In[ ]:




