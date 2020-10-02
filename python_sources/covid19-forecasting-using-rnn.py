#!/usr/bin/env python
# coding: utf-8

# # Description
# 
# * This is Version 2 of my Notebook "covid19_forecasting_using_RNN".
# * In previous notebook we have facing problem of getting constant prediction after some days.
# * In this notebook we will try to overcome that problem
# * What we will do in this notebook?
# 
#     1.  we have submission file of version 1. 
#     2.  Includinng previous submission file to our dataset
#     3.  we use the prediction of 1 april to 8 april for training of our model.
#     4.  predict for remaining test data.
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# # Data preparation for training

# In[ ]:


data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv',index_col='Date',parse_dates=True)


# In[ ]:


#feeling the missing values
data = data.fillna(value='empty')


# In[ ]:


data.head()


# In[ ]:


data['state_with_country'] = data['Province_State'] +'_'+ data['Country_Region']


# In[ ]:


data= data.drop(labels=['Province_State','Country_Region','Id'],axis=1)


# In[ ]:


data.tail()


# In[ ]:


state_with_country_name = list(data['state_with_country'].unique())


# In[ ]:


state_with_country_name[0:10]


# In[ ]:


#just feeling list for further use
state_i_data =[]
for i in state_with_country_name:
    state_i_data.append(i)


# In[ ]:


for i,j in enumerate(state_with_country_name):
     state_i_data[i] = data[data['state_with_country'] == j ]
     state_i_data[i] = state_i_data[i].drop(['state_with_country'],axis=1)
     


# In[ ]:


state_i_data[0].shape


# # Forecast data adding to Train data

# In[ ]:


prev_sub = pd.read_csv('/kaggle/input/covid19-forecasting-using-rnn/submission.csv')
test_whole = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv',index_col='Date',parse_dates=True)
test_whole = test_whole.fillna(value='empty')
#adding test set because both test and submission has same shape so we taking test index for prev_sub


# In[ ]:


test_whole['state_with_country'] = test_whole['Province_State'] +'_'+ test_whole['Country_Region']
test_whole.head()


# In[ ]:


prev_sub.index = test_whole.index


# In[ ]:


prev_sub['state_with_country'] = test_whole['state_with_country']


# In[ ]:


prev_sub.tail()


# In[ ]:


forecast = []

for i in state_with_country_name:
    forecast.append(i)


# In[ ]:


for i,j in enumerate(state_with_country_name):
     forecast[i] = prev_sub[prev_sub['state_with_country'] == j ]
     forecast[i] =  forecast[i].iloc[13:21]
     forecast[i] = forecast[i].drop(['ForecastId','state_with_country'],axis=1)
     state_i_data[i] = state_i_data[i].append(forecast[i])


# In[ ]:


forecast[293]


# In[ ]:


state_i_data[0].tail(10)


# In[ ]:





# # Data preparation for Test

# In[ ]:


test_whole = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv',index_col='Date',parse_dates=True)
test_whole = test_whole.fillna(value='empty')


# In[ ]:


test_whole.head()


# In[ ]:


test_whole['state_with_country'] = test_whole['Province_State'] +'_'+ test_whole['Country_Region']
test_whole.head()


# In[ ]:


state_with_country_name_for_test = list(test_whole['state_with_country'].unique())
state_with_country_name_for_test[:10]


# In[ ]:


state_i_data_for_test=[]
for i in state_with_country_name_for_test:

    state_i_data_for_test.append(i)


# In[ ]:


#we already have data till 31th march and previous predicted till 8 april

for i,j in enumerate(state_with_country_name_for_test):
     state_i_data_for_test[i] = test_whole[test_whole['state_with_country'] == j ]
     state_i_data_for_test[i] = state_i_data_for_test[i].iloc[21:]   
     state_i_data_for_test[i] = state_i_data_for_test[i].drop(['ForecastId','state_with_country','Province_State','Country_Region'],axis=1)
     state_i_data_for_test.append(state_i_data_for_test[i])


# In[ ]:


#state_i_data_for_test[45]


# In[ ]:





# # Creating Traing loop for all individual states and Countries

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


# In[ ]:


result = []
full_scaler = MinMaxScaler()


# In[ ]:


len(state_with_country_name)


# In[ ]:


for i in range(len(state_with_country_name_for_test)):
    scaled_full_data = full_scaler.fit_transform(state_i_data[i])
    length = 1 # Length of the output sequences (in number of timesteps)
    batch_size = 1
    generator = TimeseriesGenerator(scaled_full_data, scaled_full_data, length=length, batch_size=1)
    
    # define model
    model = Sequential()

    # Simple RNN layer
    model.add(LSTM(96,input_shape=(length,scaled_full_data.shape[1])))

    # Final Prediction (one neuron per feature)
    model.add(Dense(scaled_full_data.shape[1]))

    model.compile(optimizer='adam', loss='mse')
    
    model.fit_generator(generator,epochs=6)
    
    
    n_features = scaled_full_data.shape[1]
    test_predictions = []

    first_eval_batch = scaled_full_data[-length:]
    current_batch = first_eval_batch.reshape((1, length, n_features))

    for j in range(len(state_i_data_for_test[i])):
    
        # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
        current_pred = model.predict(current_batch)[0]
    
        # store prediction
        test_predictions.append(current_pred) 
    
        # update batch to now include prediction and drop first value
        current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
    
    
    true_predictions = full_scaler.inverse_transform(test_predictions)
    true_predictions = true_predictions.round()
    
    true_predictions = pd.DataFrame(data=true_predictions,columns=state_i_data[1].columns)
    result.append(true_predictions)
    
    print('count:-',i)
    
    
    
    
    


# In[ ]:





# # Visulization of prediction

# In[ ]:


#you can see prediction goes constant after some days
print('Plot of '+str(state_with_country_name[85]))
result[85].plot(figsize=(12,8))


# In[ ]:


print('Plot of '+str(state_with_country_name[134]))
result[134].plot(figsize=(12,8))


# # Creating Submission File

# In[ ]:


state_i_data[0].iloc[-21:]


# In[ ]:


prediction = pd.DataFrame(data= state_i_data[0].iloc[-21:] ,columns=['ConfirmedCases','Fatalities'])
prediction = prediction.append(result[0])


# In[ ]:


for i in range(1,len(result)):
    prediction = prediction.append(state_i_data[i].iloc[-21:])
    prediction = prediction.append(result[i])


# In[ ]:


len(prediction)


# In[ ]:


prediction.index = range(0,len(prediction))


# In[ ]:


prediction.head()


# In[ ]:


prediction.tail()


# In[ ]:





# In[ ]:


sub_format = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')


# In[ ]:


sub_format = sub_format['ForecastId']


# In[ ]:


final = pd.concat([sub_format,prediction],axis=1)


# In[ ]:


final.head()


# In[ ]:


final.to_csv('submission.csv',index=False)


# In[ ]:




