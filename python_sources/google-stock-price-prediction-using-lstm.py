#!/usr/bin/env python
# coding: utf-8

# ### Stock Market Prediction And Forecasting Using Stacked LSTM

# ### Import the Libraries

# In[ ]:


import numpy as np
import pandas as pd
from pandas_datareader import data, wb
from pandas.util.testing import assert_frame_equal
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import math
import datetime
import plotly
import cufflinks as cf
cf.go_offline()
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Set Duration

# In[ ]:


start = datetime.datetime(2015, 7, 11)
end = datetime.datetime(2020, 7, 11)


# ### Import the data using DataReader

# In[ ]:


df = data.DataReader("GOOG",'yahoo',start,end)
df.head()


# In[ ]:


df.tail()


# ### Exploratory Data Analysis

# #### Maximum Closing Rate

# In[ ]:


df.xs(key='Close',axis=1).max()


# #### Visualization (Closing Rate)

# In[ ]:


df.xs(key='Close',axis=1).iplot()


# #### 30-day Moving Average for Close Price

# In[ ]:


plt.figure(figsize=(12,5))
df['Close'].loc['2019-07-10':'2020-07-10'].rolling(window=30).mean().plot(label='30 Day Moving Avg.')
df['Close'].loc['2019-07-10':'2020-07-10'].plot(label='Close')
plt.legend()


# In[ ]:


df0 = df[['Open','High','Low','Close']].loc['2019-07-10':'2020-07-10']
df0.iplot(kind='candle')


# In[ ]:


df['Close'].loc['2019-07-10':'2020-07-10'].ta_plot(study='sma',periods=[9,18,27])


# #### Let's Reset the Index to Close

# In[ ]:


df1=df.reset_index()['Close']


# In[ ]:


df1


# #### Using MinMaxScaler

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[ ]:


print(df1)


# #### Splitting the Close data into Train and Test sets

# In[ ]:


training_size=int(len(df1)*0.70)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[ ]:


training_size,test_size


# In[ ]:


train_data


# In[ ]:


# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)


# In[ ]:


# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)


# In[ ]:


print(X_train.shape), print(y_train.shape)


# In[ ]:


print(X_test.shape), print(y_test.shape)


# In[ ]:


# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# ### Stacked LSTM Model

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[ ]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[ ]:


model.summary()


# In[ ]:


model.summary()


# In[ ]:


model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=64,verbose=1)


# ### Lets Predict

# In[ ]:


train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[ ]:


# Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[ ]:


### Calculate RMSE performance metrics
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[ ]:


### Test Data RMSE
math.sqrt(mean_squared_error(y_test,test_predict))


# ### Let's Visualize the Predictions

# In[ ]:


# shift train predictions for plotting
look_back=100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[ ]:


len(test_data)


# In[ ]:


x_input=test_data[278:].reshape(1,-1)
x_input.shape


# In[ ]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[ ]:


temp_input


# In[ ]:


# demonstrate prediction for next 10 days
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)


# ### Predictions for Next 30 Days

# In[ ]:


day_new=np.arange(1,101)
day_pred=np.arange(101,131)


# In[ ]:


len(df1)


# In[ ]:


plt.plot(day_new,scaler.inverse_transform(df1[1159:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))


# In[ ]:


df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[1200:])


# In[ ]:


df3=scaler.inverse_transform(df3).tolist()


# In[ ]:


plt.plot(df3)

