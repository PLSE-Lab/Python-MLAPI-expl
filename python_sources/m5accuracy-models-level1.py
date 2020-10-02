#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor

import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM

import statsmodels.api as sm
import statsmodels.tsa.api as smt



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


def downcast(df):
    dtypes = df.dtypes
    cols = dtypes.index.tolist()
    types = dtypes.values.tolist()
    
    for col, typ in zip(cols, types):
        
        if 'int' in str(typ):
            if df[col].min() > np.iinfo(np.int8).min and                 df[col].max() < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            
            elif df[col].min() > np.iinfo(np.int16).min and                 df[col].max() < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
                
            elif df[col].min() > np.iinfo(np.int32).min and                 df[col].max() < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
                
            else:
                df[col] = df[col].astype(np.int64)
                
        elif 'float' in str(typ):
            if df[col].min() > np.finfo(np.float16).min and                 df[col].max() < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
                
            elif df[col].min() > np.finfo(np.float32).min and                 df[col].max() < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
                
            else:
                df[col] = df[col].astype(np.float64)
                
        elif typ == np.object:
            if col == 'date':
                df[col] = pd.to_datetime(df[col], format='%Y-%m-%d')
                
            else:
                df[col] = df[col].astype('category')
    
    return df


# In[ ]:


#sales = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')
#price_df = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")
cal_df = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")


# In[ ]:


sales = downcast(sales)


# In[ ]:


df = pd.melt(
    sales,
    id_vars=[
        'id',
        'item_id',
        'dept_id',
        'cat_id',
        'store_id',
        'state_id'
    ],
    var_name='d',
    value_name='sold')

df.head()


# 1. Estimate level1 total unit sales
# 

# In[ ]:


df["d"] = df["d"].apply(lambda x: x.split('_')[1]).astype(np.int16)


# In[ ]:


df_level_1=pd.DataFrame(df.groupby(["d"])['sold'].sum())


# In[ ]:


from datetime import date

df_level_1.set_index(pd.to_datetime(pd.date_range(date(2011, 1,29), periods=1913).tolist()), inplace=True, drop=True)


# In[ ]:


df_level_1 = df_level_1.rename({'sold': 'sales'}, axis=1)


# In[ ]:


os.remove("/kaggle/input/m5uploaded/level_1_diff.pkl")


# In[ ]:


df_level_1=pd.DataFrame()
with open('/kaggle/input/m5upload/m5_sales_agg_level_1.pkl', 'rb') as f:
        df_level_1= pickle.load(f)


# In[ ]:



with open('/kaggle/input/m5upload/m5_level_1_diff.pkl', 'rb') as f:
        df_level_1_diff= pickle.load(f)


# In[ ]:





# 
# There are two ways you can check the stationarity of a time series. The first is by looking at the data. By visualizing the data it should be easy to identify a changing mean or variation in the data. 
# 
# For a more accurate assessment there is the Dickey-Fuller test.

# In[ ]:



#with open('/kaggle/working/m5uploaded/level_1_diff.pkl', 'wb') as f:
#        pickle.dump(df_level_1, f)


# In[ ]:


def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=5).mean()
    rolstd = timeseries.rolling(window=5).std()
    
    #Plot rolling statistics:
    
    plt.figure(figsize=(25,10))
    
    orig = plt.plot(timeseries, color='c',label='Original')
    mean = plt.plot(rolmean, color='g', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    
    
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)


# In[ ]:


plt.figure(figsize=(25,10))

plt.plot(df_level_1.index.values, df_level_1.values, linestyle='solid')
plt.xlabel('Day')
plt.ylabel('Unit Sold')
plt.title('Unit Sold Over time')


# In[ ]:


#trending, visually mean is not constant, p-value bigger than 10%, Dickey-Fuller Test fails to reject null hypothesis, series has a unit root, is non-stationary
test_stationarity(df_level_1)


# In[ ]:


from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(df_level_1, period=365)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(figsize=(40,30))
plt.subplot(411)
plt.plot(df_level_1, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()


# The yearly pattern is very obvious. and also we can see a upwards trend. Which means this data is not stationary.
# 
# To get a stationary data, there's many techiniques. We use first differencing here.

# In[ ]:


#first level differencing 
df_level_1_diff = df_level_1-df_level_1.shift()
df_level_1_diff.dropna(inplace=True)

df_level_1_diff = df_level_1_diff.rename({'sales': 'sales_diff'}, axis=1)


# In[ ]:


test_stationarity(df_level_1_diff)


#  p-value is much smaller than 1%, we can reject null hypothesis. The data after first difference is stationary.
#  
#  From the autocorrelation plot we can tell whether or not we need to add MA terms. 
#  
#  From the partial autocorrelation plot we know we need to add AR terms.

# 

# In[ ]:


#from statsmodels.graphics.tsaplots import plot_acf
#from statsmodels.graphics.tsaplots import plot_pacf
#plot_acf(df_level_1)
#plot_pacf(df_level_1, lags=50)


# In[ ]:


import statsmodels.api as sm

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df_level_1.sales, lags=40, ax=ax1) # 
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df_level_1.sales, lags=40, ax=ax2)# , lags=40


# In[ ]:


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df_level_1_diff, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df_level_1_diff, lags=40, ax=ax2)


# In[ ]:


#with open('/kaggle/working/level_1_diff.pkl', 'wb') as f:
#        pickle.dump(df_level_1_diff, f)


# Here we can see the acf and pacf both has a recurring pattern every 7 periods. Indicating a weekly pattern exists. 
# 
# We should start to consider SARIMA to take seasonality into account

# We are generating data for regressive models. Lag 7 periods as features, later will add events. 

# In[ ]:


def generate_supervised(data):
    """Generates a dataframe where each row represents a day and columns
    include sales, the dependent variable, and prior sales for each lag 7 lag features are generated. Data is used for regression modeling.
    Output df:
    d1  sales_diff  lag1  lag2  lag3 ... lag7 
    d2  sales_diff  lag1  lag2  lag3 ... lag7 
    """
    supervised_df = data.copy()

    #create column for each lag
    for i in range(1,8):
        col_name = 'lag_' + str(i)
        supervised_df[col_name] = supervised_df['sales_diff'].shift(i)

    #drop null values
    supervised_df = supervised_df.dropna().reset_index(drop=True)
    return supervised_df
    


# In[ ]:



reg_model_df = generate_supervised(df_level_1_diff)


# In[ ]:


with open('/kaggle/working/regmodel_df.pkl', 'wb') as f:
        pickle.dump(reg_model_df, f)


# In[ ]:


def tts(data):
    """Splits the data into train and test. Test set consists of the last 28
    days of data.
    """
    #data = data.drop(['sales', 'date'], axis=1)
    train, test = data[0:-28], data[-28:]

    return train, test


# In[ ]:


def scale_data(train_set, test_set):
    """Scales data using MinMaxScaler and separates data into X_train, y_train,
    X_test, and y_test.
    Keyword Arguments:
    -- train_set: dataset used to train the model
    -- test_set: dataset used to test the model
    """

    #apply Min Max Scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train_set)

    # reshape training set

    train_set = train_set.values.reshape(train_set.shape[0], train_set.shape[1])
    train_set_scaled = scaler.transform(train_set)

    # reshape test set
    test_set = test_set.values.reshape(test_set.shape[0], test_set.shape[1])
    test_set_scaled = scaler.transform(test_set)

    X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1].ravel()
    X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1].ravel()

    return X_train, y_train, X_test, y_test, scaler


# In[ ]:


def undo_scaling(y_pred, x_test, scaler_obj, lstm=False):
    """For visualizing and comparing results, undoes the scaling effect on
    predictions.
    Keyword arguments:
    -- y_pred: model predictions
    -- x_test: features from the test set used for predictions
    -- scaler_obj: the scaler objects used for min-max scaling
    -- lstm: indicate if the model run is the lstm. If True, additional
             transformation occurs
    """

    #reshape y_pred
    y_pred = y_pred.reshape(y_pred.shape[0], 1, 1)

    if not lstm:
        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

    #rebuild test set for inverse transform
    pred_test_set = []
    for index in range(0, len(y_pred)):
        pred_test_set.append(np.concatenate([y_pred[index], x_test[index]],
                                            axis=1))

    #reshape pred_test_set
    pred_test_set = np.array(pred_test_set)
    pred_test_set = pred_test_set.reshape(pred_test_set.shape[0],
                                          pred_test_set.shape[2])

    #inverse transform
    pred_test_set_inverted = scaler_obj.inverse_transform(pred_test_set)

    return pred_test_set_inverted


# In[ ]:


def predict_df(unscaled_predictions, original_df, is_arima=False):
    """Generates a dataframe that shows the predicted sales for each day
    for plotting results.
    Keyword arguments:
    -- unscaled_predictions: the model predictions that do not have min-max or
                             other scaling applied
    -- original_df: the original daily sales dataframe
    """
    
    #create dataframe that shows the predicted sales
    result_list = []
    #sales_dates = list(original_df[-29:].date)
    sales_dates = list(original_df[-29:].index)
    
    act_sales = list(original_df[-29:].sales)

    for index in range(0, len(unscaled_predictions)):
        result_dict = {}
        
        if not is_arima:
            result_dict['pred_value'] = int(unscaled_predictions[index][0] +
                                        act_sales[index])
        else:
            result_dict['pred_value'] = int(unscaled_predictions.iloc[index][0] +
                                        act_sales[index])
            
        result_dict['date'] = sales_dates[index+1]
        result_list.append(result_dict)

    df_result = pd.DataFrame(result_list)
    df_result.set_index(["date"],drop=True, inplace=True)
    return df_result


# In[ ]:


model_scores = {}

def get_scores(unscaled_df, original_df, model_name):
    """Prints the root mean squared error, mean absolute error, and r2 scores
    for each model. Saves all results in a model_scores dictionary for
    comparison.
    Keyword arguments:
    -- unscaled_predictions: the model predictions that do not have min-max or
                             other scaling applied
    -- original_df: the original daily sales dataframe
    -- model_name: the name that will be used to store model scores
    """
    rmse = np.sqrt(mean_squared_error(original_df.sales[-28:], unscaled_df.pred_value[-28:]))
    mae = mean_absolute_error(original_df.sales[-28:], unscaled_df.pred_value[-28:])
    r2 = r2_score(original_df.sales[-28:], unscaled_df.pred_value[-28:])

    rmsse_score = rmsse(np.array(ground_truth), np.array(unscaled_df), np.array(train_series), 0)
    
    model_scores[model_name] = [rmse, mae, r2, rmsse_score]

    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2 Score: {r2}")
    print(f"rmsse: {rmsse_score}")


# In[ ]:


def plot_results(results, original_df, model_name):
    """Plots predictions over original data to visualize results. Saves each
    plot as a png.
    Keyword arguments:
    -- results: a dataframe with unscaled predictions
    -- original_df: the original daily sales dataframe
    -- model_name: the name that will be used in the plot title
    """
    fig, ax = plt.subplots(figsize=(25, 7))

    original_df_plot = original_df.iloc[-28:, :]
    sns.lineplot(original_df_plot.index, original_df_plot.sales, data=original_df_plot, ax=ax,
                 label='Original', color='mediumblue')

    sns.lineplot(results.index, results.pred_value, data=results, ax=ax,
                 label='Predicted', color='red')
    ax.set(xlabel="Date",
           ylabel="Sales",
           title=f"{model_name} Sales Forecasting Prediction")
    ax.legend()
    sns.despine()

    plt.savefig(f'/kaggle/working/{model_name}_forecast.png')


# In[ ]:


def regressive_model(train_data, test_data, model, model_name, original_df):
    
    # Call helper functions to create X & y and scale data
    X_train, y_train, X_test, y_test, scaler_object =  scale_data(train_data, test_data)
    
    # Run regression model
    mod = model
    mod.fit(X_train, y_train)
    predictions = mod.predict(X_test)
    # Call helper functions to undo scaling & create prediction df
    unscaled = undo_scaling(predictions, X_test, scaler_object)
    unscaled_df = predict_df(unscaled, original_df, False)
    # Call helper functions to print scores and plot results
    get_scores(unscaled_df, original_df, model_name)
    plot_results(unscaled_df, original_df, model_name)


# In[ ]:


h = 28
n = 1885
def rmsse(ground_truth, forecast, train_series, axis=1):
    # assuming input are numpy array or matrices
    assert axis == 0 or axis == 1
    assert type(ground_truth) == np.ndarray and type(forecast) == np.ndarray and type(train_series) == np.ndarray
    
    if axis == 1:
        # using axis == 1 we must guarantee these are matrices and not arrays
        assert ground_truth.shape[1] > 1 and forecast.shape[1] > 1 and train_series.shape[1] > 1
    
    numerator = ((ground_truth - forecast)**2).sum(axis=axis)
    if axis == 1:
        denominator = 1/(n-1) * ((train_series[:, 1:] - train_series[:, :-1]) ** 2).sum(axis=axis)
    else:
        denominator = 1/(n-1) * ((train_series[1:] - train_series[:-1]) ** 2).sum(axis=axis)
    return (1/h * numerator/denominator) ** 0.5


# In[ ]:



#forecast = df_level_1_diff.iloc[-28:, 1:]
#rmsse_level_1 = rmsse(np.array(ground_truth), np.array(forecast), np.array(train_series), 0)


# In[ ]:


# Separate data into train and test sets
train, test = tts(reg_model_df)
train_series = df_level_1[:-28]
ground_truth = df_level_1[-28:]


# In[ ]:


#print(regmodel_df.shape) (1905, 8), 7 lags and 1 diff remove 8 rows from 1903  
#print(train.shape) (1877, 8)
#print(test.shape)(28, 8)
regressive_model(train, test, LinearRegression(), 'LinearRegression', df_level_1)


# In[ ]:


regressive_model(train, test, RandomForestRegressor(n_estimators=100, max_depth=20), 
          'RandomForest', df_level_1)


# In[ ]:


regressive_model(train, test, XGBRegressor( n_estimators=100, 
                                    learning_rate=0.2, 
                                    objective='reg:squarederror'), 'XGBoost', df_level_1)


# In[ ]:


def lstm_model(train_data, test_data, epochs, original_df):
    
    X_train, y_train, X_test, y_test, scaler_object = scale_data(train_data, test_data)
    
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
   
    model = Sequential()
    model.add(LSTM(4, batch_input_shape=(1, X_train.shape[1], X_train.shape[2]), 
                   stateful=True))
    model.add(Dense(1))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=epochs, batch_size=1, verbose=1, 
              shuffle=False)
    predictions = model.predict(X_test,batch_size=1)
    
    unscaled = undo_scaling(predictions, X_test, scaler_object, lstm=True)
    unscaled_df = predict_df(unscaled, original_df)
    
    get_scores(unscaled_df, original_df, 'LSTM'+str(epochs))
    
    plot_results(unscaled_df, original_df, 'LSTM'+str(epochs))


# In[ ]:


lstm_model(train, test, 2, df_level_1)


# In[ ]:


lstm_model(train, test, 50, df_level_1)


# In[ ]:


def sarimax_model(data, original_df, count):
    # Model  
    if count==1:
        sar = sm.tsa.statespace.SARIMAX(data.sales_diff, order=(1, 0, 0), seasonal_order=(1, 1, 1, 7), trend='c').fit()
    elif count==2:
        sar = sm.tsa.statespace.SARIMAX(data.sales_diff, order=(7, 0, 0), seasonal_order=(0, 1, 0, 7), trend='c').fit()
    elif count==3:
        sar = sm.tsa.statespace.SARIMAX(data.sales_diff, order=(1, 0, 1), seasonal_order=(2, 1, 2, 7), trend='c').fit()
    elif count==4:
        sar = sm.tsa.statespace.SARIMAX(data.sales_diff, order=(1, 0, 1), seasonal_order=(3, 1, 3, 7), trend='c').fit()    
    elif count==5:
        sar = sm.tsa.statespace.SARIMAX(data.sales_diff, order=(1, 0, 1), seasonal_order=(4, 2, 4, 7), trend='c').fit()
    elif count==6:
        sar = sm.tsa.statespace.SARIMAX(data.sales_diff, order=(1, 0, 1), seasonal_order=(5, 2, 5, 7), trend='c').fit()
    elif count==7:
        sar = sm.tsa.statespace.SARIMAX(data.sales_diff, order=(2, 0, 2), seasonal_order=(4, 2, 4, 7), trend='c').fit()
        
    # Generate predictions    
    start, end, dynamic = 1884, 1912, 7    
    data['pred_value'] = sar.predict(start=start, end=end)     
    # Call helper functions to undo scaling & create prediction df   
    
    
    unscaled_df = predict_df(data.iloc[-28:, 1:], original_df, is_arima=True)
    #print(unscaled_df)
    
    # Call helper functions to print scores and plot results   
    get_scores(unscaled_df, original_df, 'SARIMA'+str(count)) 
    plot_results(unscaled_df, original_df, 'SARIMA'+str(count))


# In[ ]:


#order=(1, 0, 0), seasonal_order=(1, 1, 1, 7),
sarimax_model(df_level_1_diff,  df_level_1, 1)  


# In[ ]:


#order=(7, 0, 0), seasonal_order=(0, 1, 0, 7),
sarimax_model(df_level_1_diff,  df_level_1, 2)  


# In[ ]:


#order=(1, 0, 1), seasonal_order=(2, 1, 2, 7),
sarimax_model(df_level_1_diff,  df_level_1, 3) 


# In[ ]:


sarimax_model(df_level_1_diff,  df_level_1, 4) 


# In[ ]:


sarimax_model(df_level_1_diff,  df_level_1, 5) 


# In[ ]:


sarimax_model(df_level_1_diff,  df_level_1, 6) 


# In[ ]:


sarimax_model(df_level_1_diff,  df_level_1, 7) 


# In[ ]:


pickle.dump(model_scores, open( "model_scores.p", "wb" ) )


# In[ ]:


def create_results_df():
    results_dict = pickle.load(open("/kaggle/working/model_scores.p", "rb"))
    #print(results_dict)
    #results_dict.update(pickle.load(open("/kaggle/working/arima_model_scores.p", "rb")))
    
    restults_df = pd.DataFrame.from_dict(results_dict, orient='index', 
                                        columns=['RMSE', 'MAE','R2', 'RMSSE'])
    
    restults_df = restults_df.sort_values(by='RMSE', ascending=False).reset_index()
    
    return restults_df


# In[ ]:


results = create_results_df()
results


# In[ ]:





# ToDo:
# 1. grid search SARIMA for p, d, q
# 2. add external varialbes to SARIMA
# 
# ext_var_list=['wday', 'month', 'year','event_name_1', 'event_type_1', 'event_name_2', 'event_type_2','snap_CA', 'snap_TX', 'snap_WI']
