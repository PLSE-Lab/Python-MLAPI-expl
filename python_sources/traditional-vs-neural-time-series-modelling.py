#!/usr/bin/env python
# coding: utf-8

# Hello there. In this kernel, I have used ARIMA model and Prophet for time series modelling. Then I went on to use XGBoost, followed by LSTM. I hope you enjoy this kernel.** Please upvote if you like the kernel**. Your upvotes will motivate me to code more. 

# **Importing modules**

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose 
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import make_scorer 
import math
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split,GridSearchCV
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM,Dropout,Dense,Input
from keras.models import Sequential,Model
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
import os
import xgboost as xgb
from fbprophet import Prophet
from keras import backend
print(os.listdir("../input"))


# In[ ]:


data=pd.read_csv('../input/GOOGL_2006-01-01_to_2018-01-01.csv',parse_dates=['Date'],index_col='Date')


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


data.info()


# In[ ]:


plt.subplots(2,2,figsize=(10,10))
plt.subplot(2,2,1)
data['Open'].plot()
plt.title('Open')
plt.subplot(2,2,2)
data.Close.plot()
plt.title('Close')
plt.subplot(2,2,3)
data.High.plot()
plt.title('High')
plt.subplot(2,2,4)
data.Low.plot()
plt.title('Low')
plt.tight_layout()
plt.show()


# I shall create model for data frequency of 1 Month. I have chosen the feature 'LOW'. There is no particular reason for choosing frequency of 1 Month and the 'LOW' feature.

# In[ ]:


#Let us create model for predicting Low. 

low=data['Low'].asfreq('M')
low.dropna(inplace=True)


# Let us check whether the time series is stationary or not. Stationary time series means constant mean and constant standard deviation and autocovariance that does not depend on time. So, we can check whether mean and standard deviation is constant.

# In[ ]:


#Using rolling mean and rolling standard deviation.
# I am using 'Low' values from previous 7 days (1 week) for rolling mean

def plot_rolling(data,window=7):
    rolling_mean=data.rolling(window).mean()
    rolling_std=data.rolling(window).std()
    plt.figure(figsize=(10,5))
    plt.plot(data,label='original',color='red')
    plt.plot(rolling_mean,label='rolling mean',color='black')
    plt.plot(rolling_std,label='rolling std',color='green')
    plt.legend(loc='best')
    plt.show()


# In[ ]:


plot_rolling(low)


# Stationarity can also be checked with Augmented Dickey Fuller test.  

# In[ ]:


#Lets check stationarity using Augmented Dickey-Fuller test

def test_stationarity(data):
    result=adfuller(data)
    print('ADF : ' + str(result[0]))
    print('pvalue : ' + str(result[1]))
    print('Number of lags used : ' + str(result[2]))
    print('Number of obs used : ' + str(result[3]))
    print('Critical value at 1% :' + str(result[4]['1%']))
    print('Critical value at 5% :' + str(result[4]['5%']))
    print('Critical value at 10% :' + str(result[4]['10%']))


# In[ ]:


test_stationarity(low)


# We see that the ADF value is much higher than even Critival value at 10%. This is means there is not even 90% chance that this is stationary time series.

# 2 methods to make time series stationary. First, you can subtract the rolling mean of time series from the the time series itself. Second, you can shift the time series and subtract it from the original time series itself. 

# In[ ]:


#Rolling mean subtraction from original time series
window=2
low_rolling=low.rolling(window).mean()
low_rolling_diff=low-low_rolling
low_rolling_diff=low_rolling_diff.dropna()

test_stationarity(low_rolling_diff)
plot_rolling(low_rolling_diff)


# ADF value is lesser than Critical value at 1%. That means we are more than 99% sure that this is stationary time series. Even its rolling mean and rolling std are more or less constant

# In[ ]:


# Subtraction of shifted time series from original time series
shift=1
low_shifted=low.shift(shift)
low_shift_diff=low-low_shifted
low_shift_diff=low_shift_diff.dropna()
test_stationarity(low_shift_diff)
plot_rolling(low_shift_diff)


# Similar result is obtained. You can use any one of the methods, just make sure that the time series thus obtained is stationary.

# We will use ARIMA model for prediction. 
# * **AR: Autoregression**. A model that uses the dependent relationship between an observation and some number of lagged observations.
# * **I: Integrated**. The use of differencing of raw observations (e.g. subtracting an observation from an observation at the previous time step) in order to make the time series stationary.
# * **MA: Moving Average**. A model that uses the dependency between an observation and a residual error from a moving average model applied to lagged observations.
# ARIMA has three parameters (p,d,q).
# * p: The number of lag observations included in the model, also called the lag order.
# * d: The number of times that the raw observations are differenced, also called the degree of differencing.
# * q: The size of the moving average window, also called the order of moving average.
# 
# Source : https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
# 
# I will follow you through a method to choose (p,d,q)

# In[ ]:


def plot_acf_pacf(data,lags=50):
    plot_acf(low_shift_diff,lags=lags)
    plot_pacf(low_shift_diff,lags=lags)
    plt.show()  


# Correlation represents the strength of relationship between 2 variables. We are using Pearson's correlation coefficient to determine correlation. It ranges from -1 to 1. 0 signifies no correlation. Autocorrelation is correlation between an observation in time series and a lagged observation in the same time series. Autocorrelation takes into account the direct correlations as well as indirect correlations due to the intervening observations.Partial Correlation function does not take indirect correlations into account and we can know direct correlation between two observations in time series. Source : https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/

# In[ ]:


plot_acf_pacf(low_shift_diff,lags=10)


# If value of correlation coefficient is above the red band, then it is considered to be significant. The value of p is lag value just after which the autocorrelations becomes insignificant for the first time. The value of q is lag value just after which the partial autocorrelations becomes insignificant for the first time. So, here p=2 and q=2. This is not a hard and fast rule. You can play with values of p and q around 2 and see what changes occur. 
# 

# In[ ]:


def fit(data,d=0,p=2,q=2):
    model=ARIMA(data,(p,d,q))
    model_fit=model.fit(disp=0)
    fitted_values=model_fit.fittedvalues
    score=math.sqrt(mean_squared_error(data,fitted_values))
    return fitted_values,score
        
def plot_values(predictions,data,score):
    plt.figure(figsize=(10,5))
    plt.plot(data,label='original')
    plt.plot(predictions,label='Fitted values',color='black')
    plt.title('RMSE : '+str(score))
    plt.legend(loc='best')
    plt.show()  


# **ARIMA MODEL**

# In[ ]:


fitted_values,score=fit(low_shift_diff)
plot_values(fitted_values,low_shift_diff,score)


# Now we have to convert it in original scale. We can convert so by cumulative addition of fitted_vallues followed by addition of whole series to another series of base number.

# In[ ]:


def original_scale(fitted_values,data):
    fitted_values_cumsum=fitted_values.cumsum()
    new_series=pd.Series(data[0],index=data.index)
    original_scale_fit=new_series.add(fitted_values_cumsum,fill_value=0)
    return original_scale_fit


# In[ ]:


original_scale_fit=original_scale(fitted_values,low)
original_score=math.sqrt(mean_squared_error(original_scale_fit,low))
plot_values(original_scale_fit,low,original_score)


# **AR MODEL**

# In[ ]:


fitted_values_ar,score_ar=fit(low_shift_diff,p=2,d=0,q=0)
plot_values(fitted_values_ar,low_shift_diff,score_ar)


# In[ ]:


original_scale_fit_ar=original_scale(fitted_values_ar,low)
original_score_ar=math.sqrt(mean_squared_error(original_scale_fit_ar,low))
plot_values(original_scale_fit_ar,low,original_score_ar)


# **MA MODEL**

# In[ ]:


fitted_values_ma,score_ma=fit(low_shift_diff,p=0,d=0,q=2)
plot_values(fitted_values_ma,low_shift_diff,score_ma)


# In[ ]:


original_scale_fit_ma=original_scale(fitted_values_ma,low)
original_score_ma=math.sqrt(mean_squared_error(original_scale_fit_ma,low))
plot_values(original_scale_fit_ma,low,original_score_ma)


# **Preprocessing for XGBoost**

# In[ ]:


low_date=data['Low']
low_date=pd.DataFrame(low_date,columns=['Low'])
low_date['Day']=low_date.index.day
low_date['Month']=low_date.index.month
low_date['Year']=low_date.index.year
low_date.reset_index(drop=True,inplace=True)


# In[ ]:


train_size=0.8
train_index=int(len(low_date)*train_size)
train=low_date.iloc[:train_index,:]
val=low_date.iloc[train_index:,:]
train_X=train.drop('Low',axis=1)
train_y=train['Low']
val_X=val.drop('Low',axis=1)
val_y=val['Low']


# In[ ]:


xgb_model=XGBRegressor(random_state=3)
xgb_model.fit(train_X,train_y)
pred=xgb_model.predict(val_X)
rmse=math.sqrt(mean_squared_error(val_y,pred))
val_size=val_X.shape[0]
print('RMSE for validation : '+ str(rmse/val_size)) #Normarlized with the length of validation set


# In[ ]:


train_pred=xgb_model.predict(train_X)
train_rmse=math.sqrt(mean_squared_error(train_y,train_pred))
train_size=train_X.shape[0]
print('RMSE for train : ' + str(train_rmse/train_size)) #Normalized with the length of train set


# We can see that the model is overfitting. You can use lower learning rate, tune number of estimators and use gamma for regularization to reduce overfitting. I am using GridSearch to find best parameters. I am not sure whether this approach is correct or not because in GridSearch, we do crossvalidation and so, the order in the data is lost. Please provide your suggestions in the comments. 

# In[ ]:


xgb1=XGBRegressor(random_state=5)
params={'n_estimators':np.arange(100,600,100),
       'learning_rate':np.arange(0.01,0.11,0.03),
       'gamma':np.arange(0,11,2),
       'subsample':[0.8]}

grid=GridSearchCV(xgb1,params,cv=5,scoring='neg_mean_squared_error',verbose=0)
grid.fit(train_X,train_y)


# In[ ]:


print(grid.best_params_)
print(grid.best_score_)


# In[ ]:


best_model=grid.best_estimator_
best_model.fit(train_X,train_y)
best_pred_train=best_model.predict(train_X)
train_rmse=math.sqrt(mean_squared_error(train_y,best_pred_train))
print('RMSE for training :' + str(train_rmse/train_size)) #Normalize with the length of train set
best_pred_val=best_model.predict(val_X)
val_rmse=math.sqrt(mean_squared_error(val_y,best_pred_val))
print('RMSE for validation :' + str(val_rmse/val_size)) #Normalize with the length of val set


# I guess it is still overfitting, but better than previous. You can try to optimize the parameters further. 

# **Prophet**

# Source : https://www.digitalocean.com/community/tutorials/a-guide-to-time-series-forecasting-with-prophet-in-python-3

# In[ ]:


low_df=pd.DataFrame(low).reset_index()
low_df=low_df.rename(columns={'Date':'ds','Low':'y'})


# In[ ]:


train_size=0.8
train_index=int(len(low_df)*train_size)
train=low_df.iloc[:train_index,:]
val=low_df.iloc[train_index:,:]
val_X=val.drop('y',axis=1)
val_y=val['y']


# In[ ]:


# set the uncertainty interval to 95%
prophet_model=Prophet(interval_width=0.95)
prophet_model.fit(train)
pro_predictions=prophet_model.predict(val_X)


# In[ ]:


pro_rmse_val=math.sqrt(mean_squared_error(val_y,pro_predictions['yhat']))
print('RMSE for Prophet validation : ' + str(pro_rmse_val/len(val_y))) # #Normalize with the length of val set


# In[ ]:


prophet_model.plot(pro_predictions,uncertainty=True)
plt.show()


# In[ ]:


prediction_ts=pro_predictions.loc[:,['ds','yhat']].set_index('ds')
val_ts=val.set_index('ds')


# In[ ]:


plt.figure(figsize=(10,5))
plt.plot(val_ts,label='original')
plt.plot(prediction_ts,label='prediction',color='green')
plt.legend(loc='best')
plt.show()


# **LSTM **

# **Data preparation for LSTM**

# In[ ]:


low=data['Low'].values
low=np.array(low).reshape(-1,1)


# In[ ]:


train_size=0.8
train_index=int(len(low)*train_size)
train=low[:train_index]
val=low[train_index:]


# In[ ]:


scl=MinMaxScaler(feature_range=(0,1))
train_scl=scl.fit_transform(train)
val_scl=scl.transform(val)


# In[ ]:


def prepare_dataset(data,window=30):
    X=[]
    Y=[]
    for i in range(len(data)-window):
        dummy_X=data[i:i+window]
        dummy_y=data[window+i]
        X.append(dummy_X)
        Y.append(dummy_y)
    return np.array(X),np.array(Y)

    


# In[ ]:


train_X,train_y=prepare_dataset(train_scl)
val_X,val_y=prepare_dataset(val_scl)


# In[ ]:


def rmse(true_y,pred_y):
    return backend.sqrt(backend.mean(backend.square(true_y-pred_y),axis=1))


# In[ ]:


window=30
input_layer=Input(shape=(window,1))
x=LSTM(4)(input_layer)
output=Dense(1,activation='linear')(x)
lstm_model=Model(input_layer,output)
lstm_model.compile(loss='mean_squared_error',optimizer='adam',metrics=[rmse])


# In[ ]:


result=lstm_model.fit(x=train_X,y=train_y,epochs=200,validation_data=[val_X,val_y])


# In[ ]:


lstm_train_rmse=result.history['rmse'][-1]
lstm_val_rmse=result.history['val_rmse'][-1]
print('RMSE for train :' + str(lstm_train_rmse/len(train_y)))  #Normalize with the length of train set
print('RMSE for validation :' + str(lstm_train_rmse/len(val_y)))  #Normalize with the length of val set


# In[ ]:


plt.subplots(2,1,figsize=(10,5))
plt.subplot(2,1,1)
plt.plot(result.history['rmse'],label='training')
plt.plot(result.history['val_rmse'],label='validation',color='green')
plt.legend(loc='best')
plt.subplot(2,1,2)
plt.plot(result.history['loss'],label='training')
plt.plot(result.history['val_loss'],label='validation',color='green')
plt.legend(loc='best')


# **Please upvote if you liked the kernel.** Your suggestions are always welcomed. 
# 
# 
# Things you can do further :
# * Optimize ARIMA parameters
# * Optimize XGBoost hyperparameters
# * Update the LSTM network. 
# * Use different callbacks
