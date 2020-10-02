#!/usr/bin/env python
# coding: utf-8

# ### For this Notebook, firstly I use the origin dataset to predict the confirmed cases in Ontario with LSTM which is an improved RNN architecture. I implement it by using tensorflow and Keras which we used in assignment 3.After that, I created a dataset which includes the number of cumulative cases by symptom onset date in the region of Waterloo.Then I retrian my model and predict the number of confirmed cases in a few days. 

# ### I have shared the data file on gooledrive and feel free to use it.The file link:https://drive.google.com/file/d/1N1BKwjxzI-ZGPV0dOuKGV8z1bWbNFAnr/view?usp=sharing
# ### And The data was gather from https://www.regionofwaterloo.ca/en/health-and-wellness/positive-cases-in-waterloo-region.aspx and you could upgrade the data refers to that website.

# In[ ]:


import tensorflow as tf
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# In[ ]:


#explore the dataset
dataset=pd.read_csv('../input/ece657aw20asg4coronavirus/time_series_covid19_confirmed_global.csv')
dataset.columns.values


# In[ ]:


#extract the data 
ontario_data=dataset[dataset['Province/State'].isin(['Ontario'])]
ontario_data=ontario_data.drop(['Province/State','Country/Region','Lat','Long'],axis=1)
date=ontario_data.columns.values.tolist()
feature=ontario_data.values
data = { 'number':feature[0]
      }
df = pd.DataFrame(data)
df.index=date
df


# In[ ]:


#drop the zero number
df=df.drop(index=['1/22/20', '1/23/20', '1/24/20', '1/25/20'])
df


# In[ ]:


#preprocess the data(we have to normalization of data to the range 0 to 1 to fit the LSTM architecture)
data_df=df['number'].values
data_df=data_df.reshape(-1,1)
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data_df)


# In[ ]:


#create dataset function(step refers to how much former information has relationship with the label)
def create_dataset(dataset, step):
    dataX, dataY = [], []
    for i in range(len(dataset)-step-1):
        a = dataset[i:(i+step), 0]
        dataX.append(a)
        dataY.append(dataset[i + step, 0])
    return np.array(dataX), np.array(dataY)


# In[ ]:


#create the dataset and split into train and test
#For the number of train set is very small,I just pick 10% of the data to be test set.
X,Y=create_dataset(data,4)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state=42)
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
X_train.shape


# In[ ]:


#define the architecture of the model
def create_model():
  model = Sequential()
  model.add(LSTM(32,input_shape=(4,1),return_sequences=True,activation='relu'))
  model.add(LSTM(64,return_sequences=True))
  model.add(LSTM(128))
  model.add(Dense(1))
  return model


# In[ ]:


#train the model
model=create_model()
lr_reduce =tf.keras.callbacks.ReduceLROnPlateau('val_loss',patience=3,factor=0.3,min_lr=0.00001)
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae'])
model.fit(X_train,Y_train, epochs=200, batch_size=1, validation_data=(X_test, Y_test),validation_freq=10,callbacks=[lr_reduce])


# In[ ]:


#prediction
predictions=[data_df[-1]]
newdata=data.copy()

for i in range(10):
  prediction=model.predict(newdata[-4:,:].reshape(1,4,1))
  predictions.append(float(prediction*predictions[-1]))
  newdata=np.append(newdata,prediction, axis=0)


# In[ ]:


#plot
date_predict=list(range(84,84+8))
plt.plot(data_df)
plt.plot(date_predict,predictions[3:])
plt.legend(['Origin', 'Prediction'], loc='upper left')


# In[ ]:


#using the dataset which I created
dataset_waterloo=pd.read_csv('../input/time-series-covid19-confirmed-waterloo/time_series_covid19_confirmed_waterloo.csv')
data_df_w=dataset_waterloo['Cumulative cases'].values
data_df_w=data_df_w.reshape(-1,1)
scaler = MinMaxScaler(feature_range=(0, 1))
data_w = scaler.fit_transform(data_df_w)


# In[ ]:


#create and split
X_W,Y_W=create_dataset(data_w,4)
X_train_W,X_test_W,Y_train_W,Y_test_W=train_test_split(X_W,Y_W,test_size=1,random_state=42)
X_train_W = np.reshape(X_train_W, (X_train_W.shape[0],X_train_W.shape[1],1))
X_test_W = np.reshape(X_test_W, (X_test_W.shape[0],X_test_W.shape[1],1))
X_train_W.shape


# In[ ]:


#using the same weight to predict the data
predictions=[data_df_w[-1]]
newdata=data.copy()

for i in range(10):
  prediction=model.predict(newdata[-4:,:].reshape(1,4,1))
  predictions.append(float(prediction*predictions[-1]))
  newdata=np.append(newdata,prediction, axis=0)


# In[ ]:


#plot
date_predict=list(range(45,45+8))
plt.plot(data_df_w)
plt.plot(date_predict,predictions[3:])
plt.legend(['Origin', 'Prediction'], loc='upper left')


# In[ ]:


#retrain the model by the dataset of the region of Waterloo
model=create_model()
lr_reduce =tf.keras.callbacks.ReduceLROnPlateau('val_loss',patience=3,factor=0.3,min_lr=0.00001)
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['mae'])
model.fit(X_train_W,Y_train_W, epochs=200, batch_size=1, validation_data=(X_test_W, Y_test_W),validation_freq=10,callbacks=[lr_reduce])


# In[ ]:


#predict it
predictions=[data_df_w[-1]]
newdata=data.copy()

for i in range(10):
  prediction=model.predict(newdata[-4:,:].reshape(1,4,1))
  predictions.append(float(prediction*predictions[-1]))
  newdata=np.append(newdata,prediction, axis=0)
date_predict=list(range(45,45+8))
plt.plot(data_df_w)
plt.plot(date_predict,predictions[3:])
plt.legend(['Origin', 'Prediction'], loc='upper left')


# # Conlusion
# ## 1)We can find that using different weights to predict the same data will result different,which means if we want to predict a new data set,even though we use the same model architecture,we have to retrain the model to get the new weight which may fit our new data set.
# ## 2)From the above plot, we can find predicted by model,the confirmed cases will surpass 1400 in 10 days if we don't take any action. Maybe the number is overestimated,but we still have to stay at home and do not go out except essiential reason. 
