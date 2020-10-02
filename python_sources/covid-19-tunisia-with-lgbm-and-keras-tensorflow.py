#!/usr/bin/env python
# coding: utf-8

# ![](data:image/jpeg;base64,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)

# # Estimate the confirmed covid-19-cases in Tunisia for up to 2 weeks in future with LGBM and Keras
# * based on global (country/region-wise) developments of confirmation rates including covid_19 and sars_03 outbreaks
# * Updated daily
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import os

from matplotlib import pyplot as plt
#  option for print all columns in the dataframes
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# end for options
import copy
from datetime import datetime, timedelta
from scipy import stats

import plotly.express as px
template = 'plotly_dark'
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import warnings
warnings.filterwarnings('ignore')


# # Seeing things in a simple way
# * Visualisation

# In[ ]:


df = pd.read_excel("../input/tunisa-regions-cov19/RegioCov-19.xlsx",sheet_name="RegioTime")
df.dtypes
df2 = df.copy()
#df1.rename(columns={'DateConf':'date',
 #                         'Confirmed':'confirmed',
  #                'Gouvernorat':'region'}, inplace=True)
df2.rename(columns={'DateConf':'date',
                          'Confirmed':'confirmed',
                  'Gouvernorat':'region'}, inplace=True)

# convert date in the target format for plotting the map
df2['date'] = df2['date'].dt.strftime('%d/%m/%Y')


# In[ ]:


fig = px.scatter_mapbox(df2, 
                     lat = df2["Latitude"], lon = df2["Longitude"],
                     color="confirmed", size="confirmed", hover_name="region", 
                     range_color= [0, max(df2['confirmed'])+2],  color_continuous_scale='Bluered',
                      animation_frame="date", height = 720, size_max  = 50,zoom=5,
                     # template = template,
                     title='Spread in Tunisia over time: Region')

fig.update_layout(mapbox_style="open-street-map")

fig.show()


# SARS in 2003 outbreak seems to have much in common with current COVID-19 outbreak. Hence, use additional SARS data for training.

# In[ ]:


#df_covid_19 = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv')
df_covid_19 = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")
df_covid_19['Date'] = pd.to_datetime(df_covid_19['ObservationDate'])
df_covid_19['Outbreak'] = 'COVID_2019'
df_covid_19.columns


# In[ ]:


df_sars_03 = pd.read_csv("../input/sars-outbreak-2003-complete-dataset/sars_2003_complete_dataset_clean.csv")
df_sars_03['Date'] = pd.to_datetime(df_sars_03['Date'])
df_sars_03['Province/State'] = None
df_sars_03['Outbreak'] = 'SARS_2003'
print(df_sars_03.columns)
df_sars_03.rename({'Cumulative number of case(s)':'Confirmed', 'Number of deaths':'Deaths', 'Number recovered':'Recovered', 'Country':'Country/Region'},axis=1,inplace=True)


# In[ ]:


templ_cols = ['Outbreak', 'Province/State', 'Country/Region', 'Date', 'Confirmed', 'Deaths', 'Recovered']
df = pd.concat([df_covid_19[templ_cols], df_sars_03[templ_cols]])
df = df.reset_index(drop=True)


# In[ ]:


df['Confirmed'] = df['Confirmed'].fillna(0)
df['Province/State'] = df['Province/State'].fillna('Others')
df = df.sort_values(['Country/Region','Province/State','Date'])


# In[ ]:


df = df.groupby(['Outbreak','Country/Region','Province/State','Date']).agg({'Confirmed':'sum'}).reset_index()
df['Province/State'] = 'all'


# ### Remove countries with minor confirmation numbers

# In[ ]:


t = df.groupby(['Outbreak','Country/Region','Province/State']).agg({'Confirmed':'max'})
t = t.loc[t['Confirmed'] > 2]
df = pd.merge(df,t[[]],left_on=['Outbreak','Country/Region','Province/State'], right_index=True)


# In[ ]:


df['Country/Region'].value_counts()


# In[ ]:


country_data = pd.read_csv("../input/countries-of-the-world/countries of the world.csv")
country_data['Country'] = country_data['Country'].str.strip()
country_data


# In[ ]:


df.loc[df['Country/Region']=='US','Country/Region'] = 'United States'
df.loc[df['Country/Region']=='Mainland China','Country/Region'] = 'China'
df.loc[df['Country/Region']=='Viet Nam','Country/Region'] = 'Vietnam'
df.loc[df['Country/Region']=='UK','Country/Region'] = 'United Kingdom'
df.loc[df['Country/Region']=='South Korea','Country/Region'] = 'Korea, South'
df.loc[df['Country/Region']=='Taiwan, China','Country/Region'] = 'Taiwan'
df.loc[df['Country/Region']=='Hong Kong SAR, China','Country/Region'] = 'Hong Kong'

df = pd.merge(df, country_data, how='left', left_on=['Country/Region'], right_on=['Country'])
df['Date'] = pd.to_datetime(df['Date'])


# In[ ]:


df.loc[df['Country'].isnull()]['Country/Region'].value_counts()


# In[ ]:


df.loc[df['Region'].isnull(), 'Region'] = 'Others'
df.loc[df['Country'].isnull(), 'Country'] = 'Undefined'


# In[ ]:


df['Country'].value_counts()


# In[ ]:


fix, ax = plt.subplots(figsize=(16,6), ncols=2)
s0 = df['Confirmed']
s0.plot.hist(ax=ax[0])

# BoxCox
#from sklearn.preprocessing import PowerTransformer
#transformer = PowerTransformer(method='box-cox', standardize=True)
#s0 = s0+1

# Normalise and reshape
#from sklearn.preprocessing import FunctionTransformer
#transformer = FunctionTransformer(np.log1p, inverse_func=np.expm1, validate=True)

from sklearn.preprocessing import MinMaxScaler
transformer = MinMaxScaler(feature_range=(0,1)).fit(np.asarray([0, 2E5]).reshape(-1,1)) # df['Confirmed'].values.reshape(-1,1)

s1 = pd.Series(transformer.transform(s0.values.reshape(-1,1)).reshape(-1))
s1.plot.hist(ax=ax[1])
df['Confirmed_transformed'] = s1 # make sure that every value is positive


# # Feature engineering

# In[ ]:


df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.week


# #### Lags
#   We use a 10-day-lag window to estimate the future development

# In[ ]:


n_lags = 15
for k,v in df.groupby(['Outbreak','Country/Region','Province/State']):    
    for d in range(n_lags,0,-1):                
        df.loc[v.index, f'Confirmed_Lag_{d}'] = v['Confirmed'].shift(d)
        #df.loc[v.index, f'Confirmed_Rolling_Mean_Lag{d}'] = v['Confirmed'].shift(d).rolling(n_lags).mean()
        df.loc[v.index, f'Confirmed_Transformed_Lag_{d}'] = v['Confirmed_transformed'].shift(d)

X_mask_lags = [c for c in df.columns if 'Confirmed_Lag_' in c]# + [c for c in df.columns if 'Confirmed_Rolling_Mean_Lag' in c]
X_mask_lags_transformed = [c for c in df.columns if 'Confirmed_Transformed_Lag_' in c]

df[X_mask_lags] = df[X_mask_lags].fillna(0)
df[X_mask_lags_transformed] = df[X_mask_lags_transformed].fillna(0)

print(f'Dataframe shape {df.shape}')


# # Encoding

# In[ ]:


from sklearn.preprocessing import LabelEncoder
enc_outb = LabelEncoder().fit(df['Outbreak'])
df['Outbreak_enc'] = enc_outb.transform(df['Outbreak'])

enc_ctry = LabelEncoder().fit(df['Country/Region'])
df['Country_enc'] = enc_ctry.transform(df['Country/Region'])

enc_region = LabelEncoder().fit(df['Region'])
df['Region_enc'] = enc_region.transform(df['Region'])


# # LGBM

# In[ ]:


from sklearn.model_selection import train_test_split

X_mask_cat = ['Outbreak_enc','Region_enc', 'Month','Week']
train_test = df.loc[df['Confirmed'] > 2].copy()
s_unique_values = train_test[X_mask_lags].apply(lambda r: len(np.unique(r.values)), axis=1)
train_test = train_test.loc[s_unique_values > 1].copy()
print(f'Train/Test shape {train_test.shape}')

train, valid = train_test_split(train_test, test_size=0.3, shuffle=True, random_state=231321)


# In[ ]:


from lightgbm import LGBMRegressor    
model_lgbm = LGBMRegressor(n_estimators=100000, metric='rmse', random_state=1234, min_child_samples=5, min_child_weight=0.00000001,application = 'regression'
                          , boosting = 'dart',num_leaves = 51, device = 'gpu',learning_rate = 0.003, max_bin = 63, num_iterations = 500 )

print(f'Fitting on data with shape {train[X_mask_cat+X_mask_lags].shape} with validation of shape {valid[X_mask_cat+X_mask_lags].shape}')

model_lgbm.fit(X=train[X_mask_cat+X_mask_lags], y=train['Confirmed'], 
               eval_set=(valid[X_mask_cat+X_mask_lags], valid['Confirmed']),
               early_stopping_rounds=100, verbose=5000)


# Out-of-sample prediction with 5-steps (days)

# In[ ]:


from datetime import timedelta
pred_steps = 10

history = df.loc[(df['Outbreak']=='COVID_2019') & (df['Confirmed'] > 2) & (df['Country/Region']=='Tunisia')]
history0 = history.iloc[-1]

dt_rng = pd.date_range(start=history0['Date']+timedelta(days=1), 
                       end=history0['Date']+timedelta(days=pred_steps),freq='D').values
dt_rng = pd.to_datetime(dt_rng)

pred_months = pd.Series(dt_rng).apply(lambda dt: dt.month)
pred_weeks = pd.Series(dt_rng).apply(lambda dt: dt.week)

pred_cat = history0[X_mask_cat].values
pred_lags = history0[X_mask_lags].values
y = history0['Confirmed']

print('History 0: ', pred_lags)
pred_lags[:n_lags] = np.roll(pred_lags[:n_lags], -1)
pred_lags[n_lags-1] = y  # Lag
#pred_lags[n_lags:] = np.roll(pred_lags[n_lags:], -1)
#pred_lags[-1] = np.mean(pred_lags[:n_lags]) # rolling_mean
print('Pred 0: ', pred_lags)

pred = np.zeros(pred_steps)
for d in range(pred_steps):     
    pred_cat[1] = pred_months[d]
    pred_cat[2] = pred_weeks[d]    
    
    y = model_lgbm.predict(np.hstack([pred_cat, pred_lags]).reshape(1,-1))[0]
    #print(f'Prediction body: ', np.hstack([pred_cat, pred_lags]).reshape(1,-1))
    print(f'Step {d}, predicted for {dt_rng[d].strftime("%Y-%m-%d")} is: {y}')
    
    pred_lags[:n_lags] = np.roll(pred_lags[:n_lags], -1)
    pred_lags[n_lags-1] = y  # Lag    
 #   pred_lags[n_lags:] = np.roll(pred_lags[n_lags:], -1)
 #   pred_lags[-1] = np.mean(pred_lags[n_lags:]) # rolling_mean

    pred[d] = y
    
preds = pd.Series(data=pred, index=dt_rng, name='LGBM predicted')


# # Neural network (KERAS-based)

# Use a model which is comprising some categorical informations (via embedding layers) and a LSTM structure to learn the time-dependent behavior.

# In[ ]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, TimeDistributed, RepeatVector, Input, Concatenate, Flatten, Reshape, Embedding
from tensorflow.keras.backend import clear_session
import keras
clear_session()

inp_outbreak = Input(shape=(1,1))
inp_country = Input(shape=(1,1))
inp_lags = Input(shape=(n_lags,1))

emb_outbreak = Embedding(input_dim=2, output_dim=1)(inp_outbreak)
emb_country = Embedding(input_dim=200, output_dim=2)(inp_country)

lstm1 = LSTM(64, activation='linear', return_sequences=True)(inp_lags)
lstm2 = LSTM(32, activation='linear', return_sequences=False)(lstm1)

concat1 = Reshape(target_shape=(1,3))(Concatenate(axis=3)([emb_outbreak, emb_country]))
concat2 = Concatenate(axis=1)([Flatten()(concat1), lstm2])
dense1 = Dense(32, activation='linear')(concat2)
dense2 = Dense(1, activation='linear')(dense1)

model_keras = Model(inputs=[inp_outbreak, inp_country, inp_lags], outputs=[dense2])
model_keras.compile(loss='mean_squared_error', optimizer='adam')
model_keras.summary()


def prepare_keras_input(data):
    lags = data[X_mask_lags_transformed].values.reshape(-1, 15, 1)
    y = data['Confirmed_transformed'].values.reshape(-1,1)    
    
    return [data['Outbreak_enc'].values.reshape(-1,1,1), 
            data['Country_enc'].values.reshape(-1,1,1), lags], y
    
train_X, train_y = prepare_keras_input(train)
model_keras.fit(train_X, train_y, validation_data=(prepare_keras_input(valid)), epochs=50, verbose=0,validation_steps=10,use_multiprocessing=True)


# * Out-of-sample prediction with 3-steps (days)

# In[ ]:


from datetime import timedelta
from tensorflow import convert_to_tensor

pred_steps = 10

history = df.loc[(df['Outbreak']=='COVID_2019') & (df['Confirmed'] > 2) & (df['Country/Region']=='Tunisia')]
history0 = history.iloc[-1]

pred_cat_outbreak = convert_to_tensor(history0['Outbreak_enc'].reshape(-1,1,1), np.int32)
pred_cat_country = convert_to_tensor(history0['Country_enc'].reshape(-1,1,1), np.int32)

pred_lags = history0[X_mask_lags_transformed].values
y = history0['Confirmed_transformed']

#print('History 0: ', pred_lags)
pred_lags = np.roll(pred_lags, -1)
pred_lags[-1] = y

#print('Pred 0: ', pred_lags)
pred = np.zeros(pred_steps)

dt_rng = pd.date_range(start=history0['Date']+timedelta(days=1), 
                       end=history0['Date']+timedelta(days=pred_steps),freq='D').values
dt_rng = pd.to_datetime(dt_rng)
# Scale
for d in range(pred_steps):    
    y = model_keras.predict([pred_cat_outbreak, pred_cat_country, convert_to_tensor(pred_lags.reshape(-1,15,1), np.float32)])[0][0]
    #print(f'Pred body: {pred_lags}')
    print(f'Step {d}, predicted for {dt_rng[d].strftime("%Y-%m-%d")} is: {transformer.inverse_transform(y.reshape(-1,1)).reshape(-1)}')
    
    pred_lags = np.roll(pred_lags, -1)
    pred_lags[-1] = y    
    pred[d] = y
    
pred = transformer.inverse_transform(pred.reshape(-1,1)).reshape(-1)
preds_keras = pd.Series(data=pred, index=dt_rng, name='Keras predicted')


# In[ ]:


from matplotlib import pyplot as plt
fig, ax = plt.subplots(figsize=(16,7))

hist = history.set_index(['Date'])['Confirmed'].plot(ax=ax, marker='o')
preds.plot(ax=ax, marker='o', linewidth=2)
preds_keras.plot(ax=ax, marker='*')
plt.legend()
plt.tight_layout()


# # Saving the output

# In[ ]:


import datetime
ts = datetime.datetime.now().strftime('%Y%m%d')
df_out = pd.DataFrame([preds, preds_keras]).T
df_out.to_csv(f'{ts}_prediction.csv')


# In[ ]:


from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
df_ser = df[df['Country/Region']=='Tunisia']
df_ser


# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

series = df_ser.loc[:,['Date','Confirmed']]
series['Date'] = pd.to_datetime(series['Date'])
dates = np.asarray(series['Date'])
series.reset_index()
series = df_ser.loc[:,['Confirmed']]



#conf = np.asarray(series['confirmed'])
#series = pd.Series(conf, index=dates,columns = ['confirmed'])
#series['confirmed'].values().index(dates)
#parse_dates
series = series.set_index(dates)
series


# In[ ]:


autocorrelation_plot(series)
pyplot.show()


# In[ ]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(311)
fig = plot_acf(series, ax=ax1,
               title="Autocorrelation on Original Series") 
ax2 = fig.add_subplot(312)
fig = plot_acf(series.diff().dropna(), ax=ax2, 
               title="1st Order Differencing")
ax3 = fig.add_subplot(313)
fig = plot_acf(series.diff().diff().dropna(), ax=ax3, 
               title="2nd Order Differencing")


# In[ ]:


plot_pacf(series.diff().dropna(), lags=10)


# In[ ]:


plot_acf(series.diff().dropna())


# In[ ]:


from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = ARIMA(series, order=(0,1,1))
model_fit = model.fit(start_ar_lags=1 ,transparams=False)
print(model_fit.summary())
# plot residual errors
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 10))
ax = series.loc['2020-03-04':].plot(ax=ax)
model_fit.plot_predict('2020-04-01','2020-04-12',ax=ax,plot_insample=False)
plt.show()


# In[ ]:


X = series.values
size = int(len(X) * 0.5)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = SARIMAX(history, order=(0,2,1))
    model_fit = model.fit(maxiter=200)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
   # print(model_fit.mle_retvals)
from sklearn.metrics import mean_squared_error
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)


# In[ ]:


# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()


# In[ ]:




#divide into train and validation set
train = series[:int(0.7*(len(series)))]
valid = series[int(0.7*(len(series))):]




#plotting the data
train['Confirmed'].plot()
valid['Confirmed'].plot()


# In[ ]:


get_ipython().system('pip install pyramid-arima')


# In[ ]:



#building the model
from pyramid.arima import auto_arima
model = auto_arima(train, error_action='ignore', suppress_warnings=False,seasonal=False)
model.fit(train)

forecast = model.predict(n_periods=len(valid))
forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])

#plot the predictions for validation set
plt.plot(train, label='Train')
plt.plot(valid, label='Valid')
plt.plot(forecast, label='Prediction')
plt.show()


# In[ ]:


from statsmodels.tsa.stattools import adfuller
def adf_test(timeseries):
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

#apply adf test on the series
print(adf_test(train['Confirmed']))


# In[ ]:


from math import sqrt
rms = sqrt(mean_squared_error(valid,forecast))
print(rms)


# # Bayesian Neural network

# In[ ]:


import tensorflow as tf
tfk = tf.keras
tf.keras.backend.set_floatx("float64")
import tensorflow_probability as tfp
tfd = tfp.distributions
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
# Define helper functions.
scaler = StandardScaler()
detector = IsolationForest(n_estimators=1000, behaviour="deprecated", contamination="auto", random_state=0)
neg_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)


# In[ ]:


data = series
# Scale data to zero mean and unit variance.
X_t = scaler.fit_transform(data)


# In[ ]:


# Restore frame.
dataset = pd.DataFrame(X_t, columns =['Confirmed'])
dataset = dataset.set_index(dates)
dataset


# In[ ]:


# Define some hyperparameters.
n_epochs = 50
n_samples = dataset.shape[0]
n_batches = 10
batch_size = np.floor(n_samples/n_batches)
buffer_size = n_samples
# Define training and test data sizes.
n_train = int(0.7*dataset.shape[0])

# Define dataset instance.
data = tf.data.Dataset.from_tensor_slices((dataset['Confirmed'].values))
data = data.shuffle(n_samples, reshuffle_each_iteration=True)

# Define train and test data instances.
data_train = data.take(n_train).batch(batch_size).repeat(n_epochs)
data_test = data.skip(n_train).batch(1).repeat(n_epochs)

