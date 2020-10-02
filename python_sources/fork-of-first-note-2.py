#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from math import sqrt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample = pd.read_csv('../input/sample_submission.csv')

train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])

train['month'] = train['date'].dt.month
train['day'] = train['date'].dt.dayofweek
train['year'] = train['date'].dt.year

test['month'] = test['date'].dt.month
test['day'] = test['date'].dt.dayofweek
test['year'] = test['date'].dt.year

train['block'] = 60-((train['year']-2013)*12+train['month'])
train['week'] = 260-((train['year']-2013)*52+train['date'].dt.week)


# In[ ]:


# Grouping by Date/Time to calculate number of trips
day_df = pd.Series(train.groupby(['date'])['sales'].sum())
# setting Date/Time as index
day_df.index = pd.DatetimeIndex(day_df.index)
# Resampling to daily trips
day_df = day_df.resample('1D').apply(np.sum)

day_df.plot()


# # trend - autocorrelation
# 
# * there is a upward trend
# * there is a 7day  week autocorrelation effect
# * Yearly effects
# ** there is a summer peak  effect
# ** there is a 'halloween season' 
# 

# In[ ]:


#Checking trend and autocorrelation
def initial_plots(time_series, num_lag):

    #Original timeseries plot
    plt.figure(1)
    plt.plot(time_series)
    plt.title('Original data across time')
    plt.figure(2)
    plot_acf(time_series, lags = num_lag)
    plt.title('Autocorrelation plot')
    plot_pacf(time_series, lags = num_lag)
    plt.title('Partial autocorrelation plot')
    
    plt.show()

    
#Augmented Dickey-Fuller test for stationarity
#checking p-value
print('p-value: {}'.format(adfuller(day_df)[1]))

#plotting
initial_plots(day_df, 45)


# # differencing
# 
# trend disappears as expected
# 7day week effect remains

# In[ ]:


#storing differenced series
diff_series = day_df.diff(periods=1).diff(periods=7)

#Augmented Dickey-Fuller test for stationarity
#checking p-value
print('p-value: {}'.format(adfuller(diff_series.dropna())[1]))
initial_plots(diff_series.dropna(), 30)


# # week arima
# 
# * week effect disappears
# * differencing there is a year effect left over
# 

# In[ ]:


# Grouping by Date/Time to calculate number of trips
week_df = pd.Series(train.groupby(['week'])['sales'].sum())
# setting Date/Time as index
week_df.plot()
initial_plots(week_df, 45)


# In[ ]:


#storing differenced series
diff_series = week_df.diff(periods=1)

#Augmented Dickey-Fuller test for stationarity
#checking p-value
print('p-value: {}'.format(adfuller(diff_series.dropna())[1]))
initial_plots(diff_series.dropna(), 55)


# # month
# * year cycle very visible
# * differencing 1 month makes everything right
# 

# In[ ]:


# Grouping by Date/Time to calculate number of trips
month_df = pd.Series(train.groupby(['block'])['sales'].sum())
# setting Date/Time as index
month_df.plot()
initial_plots(month_df, 45)


# In[ ]:


#storing differenced series
diff_series = month_df.diff(periods=1).diff(periods=12)
#Augmented Dickey-Fuller test for stationarity
#checking p-value
print('p-value: {}'.format(adfuller(diff_series.dropna())[1]))
initial_plots(diff_series.dropna(), 30)


# # a forecast brute

# In[ ]:


train_p=train.pivot_table(index=['store', 'item'], columns='date', values='sales')
train_p.head()


# In[ ]:


def piv_clust(data,veld,kolom,waarde,komponent):
    from sklearn.decomposition import TruncatedSVD,FastICA
    df = data.pivot_table(index=veld, columns=kolom, values=waarde, fill_value=0, aggfunc=np.sum)            #pivot table > signal should follow
    svd = TruncatedSVD(n_components=komponent, n_iter=7, random_state=42)                                    #decomp functions
    ica = FastICA(n_components=komponent, max_iter=1000, tol=0.1)
    df_norm =( (df - df.mean()) / (df.max() - df.min()) ).fillna(0)                                          #normalize
    return pd.DataFrame( np.concatenate([svd.fit_transform(df_norm)*svd.singular_values_, ica.fit_transform(df_norm)],axis=1) , index=df.index,columns=['clus'+str(x) for x in range(komponent+komponent)]),svd.explained_variance_ratio_  #U*S


item_c,item_sing=piv_clust(train,'item',['store','block'],'sales',10)
store_c,store_sing=piv_clust(train,'store',['item','block'],'sales',10)
#train_s,sing_s=piv_clust(train_df,'shop_id',['item_category_id','date_block_num'],'item_price',10)
#train_si,sing_si=piv_clust(train_df[train_df['date_block_num']>10],'item_id',['date_y'],'item_cnt_day',10)


# In[ ]:


train_t=train_p.merge(item_c,left_index=True,right_index=True)
train_t=train_t.merge(store_c,left_index=True,right_index=True)


# In[ ]:


train_t.shape


# In[ ]:


# Create linear regression object
regr = linear_model.LinearRegression()

for xi in range(26,27):
    # Train the model
    x = train_t.iloc[:,1:]
    y = train_t.iloc[:,0]
    regr.fit(x, y)
    # Make predictions
    pred = regr.predict(x)
    # The coefficients
    print('Intercept: \n', regr.intercept_)
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print(xi,'MSRE: %.4f'%np.sqrt(((y-pred)*(y-pred)).mean()))
    
    train_f=train_p.iloc[:,:-1].merge(item_c,left_index=True,right_index=True)
    train_f=train_f.merge(store_c,left_index=True,right_index=True)
    pred =regr.predict(train_f)
    print(pred)


# In[ ]:



test['block'] = 60-((test['year']-2013)*12+test['month'])
test.groupby(['store', 'item']).max()


# In[ ]:


col = [i for i in test.columns if i not in ['date','id']]
y = 'sales'
print(train_p.shape)
ytrain=train_p.iloc[:,-1:]
train_x, train_cv, y, y_cv = train_test_split(train_p.iloc[:,-1:].reset_index(),ytrain, test_size=0.2, random_state=2018)

def XGB_regressor(train_X, train_y, test_X, test_y, feature_names=None, seed_val=2017, num_rounds=300):
    param = {}
    param['objective'] = 'reg:linear'
    param['eta'] = 0.2
    param['max_depth'] = 12
    param['silent'] = 0
    param['eval_metric'] = 'mae'
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())

    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=10)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)
        
    return model    
    
    
model = XGB_regressor(train_X = train_x, train_y = y, test_X = train_cv, test_y = y_cv)
y_test = model.predict(xgb.DMatrix(pd.DataFrame(train_f,columns=train_p.iloc[:,-1:].columns).reset_index()), ntree_limit = model.best_ntree_limit)
print(y_test)
#sample['sales'] = y_test
#sample.to_csv('simple_xgb_starter.csv', index=False)

