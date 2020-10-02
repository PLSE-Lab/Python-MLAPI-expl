#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pylab as plt
import matplotlib.dates as mdates
plt.rcParams['figure.figsize'] = (15.0, 8.0)
import pandas as pd
import seaborn as sns

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot


# In[ ]:


#!pip install bokeh==0.9.3 -U


# In[ ]:


from pandas.tseries.holiday import USFederalHolidayCalendar as calendar


# In[ ]:


import bokeh as bk


# In[ ]:


bk.__version__


# In[ ]:


from bokeh.charts import output_file, show ,TimeSeries 
from bokeh.io import output_notebook
output_notebook()


# ### Data Preparation
# Eleven months data from one bed one bath apartment unit in San Jose, CA region was picked for this experiment. The electricity consumption is recorded in 15 minutes interval by the energy supply company. The raw data contains information such as type, date, start time, end time, usage, units, cost and notes fields. The start time and end time is the measurement interval. In this data, the interval is 15 minutes. The usage in 15 minutes interval is provided in kWh unit, and the cost for the consumption is presented in the dollar. Before we deep dive into the data, some quick feature engineering steps are done to enrich the data with more features. 

# In[ ]:


data = pd.read_csv("/kaggle/input/D202.csv")
data.head(2)


# #### Creating Date and Time Filed

# In[ ]:


data["DATE_TIME"] = pd.to_datetime(data.DATE + " " + data["END TIME"])


# #### Working Day or Not

# In[ ]:


data["DAY_TYPE"] = data.DATE_TIME.apply(lambda x: 1 if x.dayofweek > 5 else 0  )


# #### Finding Fedaral Holidays

# In[ ]:


cal = calendar()
holidays = cal.holidays(start = data.DATE_TIME.min(), end = data.DATE_TIME.max())
data["IS_HOLIDAY"] = data.DATE_TIME.isin(holidays)


# In[ ]:


data.head(3)


# #### Previous Five Observations

# In[ ]:


for obs in range(1,6):
    data["T_" + str(obs)] = data.USAGE.shift(obs)


# In[ ]:


data.fillna(0.00,inplace=True)
data.head(10)


# In[ ]:


data.IS_HOLIDAY = data.IS_HOLIDAY.astype("int")


# In[ ]:


data.head(2)


# #### Clean Data

# In[ ]:


clean_data = data[['DAY_TYPE', 'IS_HOLIDAY', 'T_1','T_2', 'T_3', 'T_4', 'T_5','USAGE']]


# In[ ]:


clean_data.head(2)


# In[ ]:


data_tmp_dict = dict(Usage=data['USAGE'], Date=data['DATE_TIME'])


# In[ ]:


p = TimeSeries(data_tmp_dict, index='Date', title="Usage", ylabel='Usage in kWh')

show(p)


# ### Let's Explore

# In[ ]:





# ### A Week ! Yes X'Mas Week

# In[ ]:


xmask = (data.DATE_TIME >= pd.to_datetime("12/20/2016")) & 
(data.DATE_TIME <= pd.to_datetime("12/27/2016"))


# In[ ]:


xmas_week = data.loc[xmask]


# In[ ]:


xmas_dict = dict(Usage=xmas_week['USAGE'],Date=xmas_week['DATE_TIME'])
#data_tmp_dict = dict(Usage=data['USAGE'], Date=data['DATE_TIME'])


# In[ ]:


xmas_show = TimeSeries(xmas_dict, index='Date', title="Usage Xmas Day", ylabel='Usage in kWh')

show(xmas_show)


# ### A Day ! New Year 2017

# In[ ]:


dmask = (data.DATE_TIME >= pd.to_datetime("01/01/2017")) & (data.DATE_TIME < pd.to_datetime("01/02/2017"))
nyd = data.loc[dmask]


# In[ ]:


nyd_dict = dict(Usage=nyd['USAGE'],Date=nyd['DATE_TIME'])


# In[ ]:


nyd_show = TimeSeries(nyd_dict, index='Date', title="Usage New Years Day", ylabel='Usage in kWh')

show(nyd_show)


# ### Train and Test Data

# In[ ]:


training_data = data[data.DATE_TIME < pd.to_datetime("08/01/2017")]


# In[ ]:


val_mask = (data.DATE_TIME >= pd.to_datetime("08/01/2017")) & (data.DATE_TIME < pd.to_datetime("09/01/2017"))
val_data = data.loc[val_mask]


# In[ ]:


test_data = data[data.DATE_TIME >= pd.to_datetime("09/01/2017")]


# In[ ]:


training_data.tail(3)


# In[ ]:


test_data.head(2)


# In[ ]:


clean_train = training_data[['DAY_TYPE', 'IS_HOLIDAY', 'T_1','T_2', 'T_3', 'T_4', 'T_5','USAGE']]
clean_test = test_data[['DAY_TYPE', 'IS_HOLIDAY', 'T_1','T_2', 'T_3', 'T_4', 'T_5','USAGE']]
clean_val = val_data[['DAY_TYPE', 'IS_HOLIDAY', 'T_1','T_2', 'T_3', 'T_4', 'T_5','USAGE']]


# In[ ]:


clean_train.head(2)


# In[ ]:


clean_test.head(2)


# In[ ]:


clean_val.head(3)


# ### Let's Model and Predict

# In[ ]:


from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error


# In[ ]:


X_train,y_train = clean_train.drop(["USAGE"],axis=1),clean_train.USAGE
X_test,y_test = clean_test.drop(["USAGE"],axis=1),clean_test.USAGE
X_val,y_val = clean_val.drop(["USAGE"],axis=1),clean_val.USAGE


# In[ ]:


scaler = StandardScaler()
#scaler = MinMaxScaler(feature_range=(-1, 1))
rfr  = RandomForestRegressor(random_state=2017,verbose=2,n_jobs=5)


# In[ ]:


X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
X_valid_scaled = scaler.fit_transform(X_val)


# In[ ]:





# In[ ]:


rfr.fit(X_train_scaled,y_train)


# In[ ]:


rfr.score(X_val,y_val)


# In[ ]:


rfr.score(X_test,y_test)


# In[ ]:


test_data["RF_PREDICTED"] = rfr.predict(X_test_scaled)


# In[ ]:


test_data.head(5)


# ### Prediction with Random Forest Model in Test Data

# In[ ]:


pred_show = TimeSeries(test_data,x="DATE_TIME",y=["USAGE","RF_PREDICTED"],legend=True,plot_width=800, plot_height=350)
show(pred_show)


# #### Prediction Single Day in Test Data

# In[ ]:


sep_30m = test_data[test_data.DATE_TIME >= pd.to_datetime("09/30/2017")]

sep_30rf = TimeSeries(sep_30m,x="DATE_TIME",y=["USAGE","RF_PREDICTED"],legend=True,plot_width=900, plot_height=350)
show(sep_30rf)


# ### LSTM Modelling

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# #### LSTM Model

# In[ ]:


model_k = Sequential()
model_k.add(LSTM(1, input_shape=(1,7)))
model_k.add(Dense(1))
model_k.compile(loss='mean_squared_error', optimizer='adam')


# In[ ]:


SVG(model_to_dot(model_k).create(prog='dot', format='svg'))


# #### Reshape the data to 3D

# In[ ]:


X_t_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))


# In[ ]:


X_val_resaped = X_valid_scaled.reshape((X_valid_scaled.shape[0], 1, X_valid_scaled.shape[1]))


# ##### Fit the Model

# In[ ]:


history = model_k.fit(X_t_reshaped, y_train, validation_data=(X_val_resaped, y_val),epochs=10, batch_size=96, verbose=2)


# In[ ]:


plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()


# In[ ]:


X_te_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))


# In[ ]:


res = model_k.predict(X_te_reshaped)


# In[ ]:


test_data["DL_PRED"] = res


# #### LSTM Prediction on Test Data

# In[ ]:


keras_show = TimeSeries(test_data,x="DATE_TIME",y=["USAGE","RF_PREDICTED","DL_PRED"],legend=True,plot_width=900, plot_height=350)
show(keras_show)


# #### A Day on LSTM Predcted Result

# In[ ]:


sep_30m = test_data[test_data.DATE_TIME >= pd.to_datetime("09/30/2017")]

sep_30 = TimeSeries(sep_30m,x="DATE_TIME",y=["USAGE","RF_PREDICTED","DL_PRED"],legend=True,plot_width=900, plot_height=350)
show(sep_30)


# #### RMSE Value of Random Forest and LSTM

# In[ ]:


from numpy import sqrt
sqrt(mean_squared_error(test_data.USAGE,test_data.DL_PRED))


# In[ ]:


sqrt(mean_squared_error(test_data.USAGE,test_data.RF_PREDICTED))

