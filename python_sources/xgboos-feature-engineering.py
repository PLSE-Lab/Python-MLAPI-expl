#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import matplotlib.pyplot as plt
from matplotlib import cm as cm
import xgboost as xgb

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split


# In[ ]:


def evaluate_model(y, y_pred):
    mae, mse, msqle, mabse, r2 = mean_absolute_error(y, y_pred),         mean_squared_error(y, y_pred),         mean_squared_log_error(y, y_pred),         median_absolute_error(y, y_pred),         r2_score(y, y_pred)
    print('mae', mae)
    print('mse', mse)
    print('msqle', msqle)
    print('mabse', mabse)
    print('r2', r2)


# In[ ]:


def predictions_to_submission_file(predictions):
    submission_df = pd.DataFrame(columns=['Expected', 'Id'])
    submission_df['Expected'] = predictions
    submission_df['Id'] = range(len(predictions))
    submission_df.to_csv('submission.csv', index=False)


# In[ ]:


def correlation_matrix(df):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Abalone Feature Correlation')
    labels=['Sex','Length','Diam','Height','Whole','Shucked','Viscera','Shell','Rings',]
    ax1.set_xticklabels(labels,fontsize=6)
    ax1.set_yticklabels(labels,fontsize=6)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
    plt.show()


# In[ ]:


def new_features(df, _n):
    for i in range(12):
        for _i in range(_n):
            df[f'sensor#{i}-delta{_i}'] = df[f'sensor#{i}'] - df[f'sensor#{i}'].shift(_i)
        for _i in range(3,_n):
            df[f'sensor#{i}-mva{_i}'] = df[f'sensor#{i}'].rolling(_i, win_type ='triang').sum()
        
    return df.fillna(0)


# ### Read data

# In[ ]:


train_df = pd.read_csv('train.csv')
train_df = new_features(train_df, 13)
train_df.head()


# In[ ]:


test_df = pd.read_csv('test.csv')
test_df = new_features(test_df, 13)
test_df.head()


# ### Select columns with sensor data

# In[ ]:


columns_in = list(train_df.columns[1:13]) + list(train_df.columns[15:291])
columns_out = "oil_extraction"

train_x = train_df[columns_in].values
train_y = train_df[columns_out].values
test_x = test_df[columns_in].values


# ### Split dataset

# In[ ]:


X_train, X_test, y_train, Y_test = train_test_split(train_x, train_y, test_size=0.3, random_state=2)


# In[ ]:


X_train.shape


# In[ ]:


gb_reg = GradientBoostingRegressor(n_estimators=400, learning_rate=0.1, max_depth=4)
gb_reg.fit(X_train, y_train)
train_predictions = gb_reg.predict(train_x)
evaluate_model(train_y, train_predictions)
print('====')
train_predictions = gb_reg.predict(X_test)
evaluate_model(Y_test, train_predictions)


# In[ ]:


xg_req = xgb.XGBRegressor(objective='reg:linear', learning_rate=0.1, max_depth=8, n_estimators=900, n_jobs=8)


# In[ ]:


xg_req.fit(X_train, y_train)
train_predictions = xg_req.predict(train_x)
evaluate_model(train_y, train_predictions)
print('====')
train_predictions = xg_req.predict(X_test)
evaluate_model(Y_test, train_predictions)


# ### Write submission file

# In[ ]:


test_predictions = xg_req.predict(test_x)
predictions_to_submission_file(test_predictions)


# In[ ]:




