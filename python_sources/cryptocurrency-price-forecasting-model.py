#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# after a complete run program  has to get stopped -- to run -- more higher performance metrics -- to solve the problems -->


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# there are lots of deliverable projects -- 
"""
whats
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# local files that could be good for the  prediction sequences --> 
# if  the training  losses and  validation losses  are particularly high when running t
# Any results you write to the current directory are saved as output.
# load with the several data flow algorithms --> with all the loadable values |-->  
# all_data


# In[ ]:



import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import requests
import json
import datetime
from datetime import datetime
from tensorflow import keras # some  required vision for the running the algorithms -- 
# linux time stamp is applied in this  time series 
# json and requests libraries needs to be imported before hand
# cryptocompare credentials for  this part of the functionis very optional to run -- so needs to load all the rest of the cases .. it 

#url = "https://min-api.cryptocompare.com/data/v2/histominute?fsym=BTC&tsym=USD&limit=10"
#apikey = "424e0c65bd9be940591fe30fae4a23fdfce9e192eaa09e991f65aad43285277b"
# run all the rest of the methods .. 
#params = {"key":apikey}
#response = requests.get(url, params)
# run all of those values that run in the sins
# form of the potential energy based on the fragility and dependency on the electronic systems .. 
# two different scaler making sure to get loaded on the data frame --. 
def get_historical_hourly():
    """
    goal is that one side stationary to predict the price values for the following  time stamp with not high precision
    
    """
    
    print("Which coins table data do you want to have?")
    #fsym = input()
    fsym = "MATIC"
    print("How many minutes of data in from the past do you want to extract?")
    #limit = input() # limit is a number of minutes that program will extract data  -->
    limit = "2000"
    url = "https://min-api.cryptocompare.com/data/v2/histohour?fsym="+fsym+"&tsym=USDT&limit="+limit
    
    apikey = "424e0c65bd9be940591fe30fae4a23fdfce9e192eaa09e991f65aad43285277b"
    params = {"key":apikey}
    response = requests.get(url, params)
    temp = response.text
    values = json.loads(temp)
    # read --> the functional and 
    
    return values["Data"]["Data"]
def get_historical_daily():
    # it is interesting to download by hourly or daily that can be newer version will be applied that can't be changed -->
    # access whatever happens to the world that is mattering --> 
    """
    
    goal is that one side stationary to predict the price values for the following  time stamp with not high precision
    
    """
    
    print("Which coins table data do you want to have?")
    #fsym = input()
    fsym = "MATIC"
    print("How many minutes of data in from the past do you want to extract?")
    #limit = input() # limit is a number of minutes that program will extract data  -->
    limit = "2000"
    url = "https://min-api.cryptocompare.com/data/v2/histominute?fsym="+fsym+"&tsym=USDT&limit="+limit
    
    apikey = "424e0c65bd9be940591fe30fae4a23fdfce9e192eaa09e991f65aad43285277b"
    params = {"key":apikey}
    response = requests.get(url, params)
    temp = response.text
    values = json.loads(temp)
    return values["Data"]["Data"]

def tabledatabuilder(data):
    """
    this method will get the collecion of the dictionary  and return the tabular data on that -->

    """
    columns = ["time", "high", "low", "volumefrom", "volumeto", "close"]
    price_minutes = pd.DataFrame(columns = ["time", "high", "low", "volumefrom", "volumeto", "close"])
    time =  []
    high_prices = []
    low_prices =  []
    volumefrom = []
    volumeto = []
    close_prices = []
    standard_time =  []

    for eachminutedata in data:
        time.append(eachminutedata["time"])
        high_prices.append(eachminutedata["high"])
        low_prices.append(eachminutedata["low"])
        volumefrom.append(eachminutedata["volumefrom"])
        volumeto.append(eachminutedata['volumeto'])
        close_prices.append(eachminutedata['close'])
        standard_time.append(datetime.fromtimestamp(eachminutedata["time"]))
    price_minutes['time'] = time
    price_minutes["high"] = high_prices
    price_minutes['low'] = low_prices
    price_minutes["volumefrom"] = volumefrom
    price_minutes['volumeto'] = volumeto
    price_minutes["close"] = close_prices
    price_minutes["standardtime"] = standard_time
    return price_minutes




def get_historical():
    """
    goal is that one side stationary to predict the price values for the following  time stamp with not high precision
    
    """
    
    print("Which coins table data do you want to have?")
    #fsym = input()
    fsym = "MATIC"
    print("How many minutes of data in from the past do you want to extract?")
    #limit = input() # limit is a number of minutes that program will extract data  -->
    limit = "2000"
    url = "https://min-api.cryptocompare.com/data/v2/histohour?fsym="+fsym+"&tsym=USDT&limit="+limit
    
    apikey = "424e0c65bd9be940591fe30fae4a23fdfce9e192eaa09e991f65aad43285277b"
    params = {"key":apikey}
    response = requests.get(url, params)
    temp = response.text
    values = json.loads(temp)
    return values["Data"]["Data"]

data = get_historical()
print(data[-1]['time'])
data = tabledatabuilder(data)
sorteddata = data.sort_values("time", ascending = False)
# can make the data that is called by  the time and day of the year --> 
data.to_csv("data_matic_hourly.csv", index = False, encoding = "utf-8")
print(data.shape)
data.head()


# In[ ]:


def add_member(np_series, value):
    """
    np_series is the 1 element -->
    adds a values from the last element - for the np.series -->  that is possible to 
    all index is shifting by 1 to the the end --> 
    """
    np_series = np_series[1:]
    np_series = list(np_series)
    np_series.append(value)
    np_series = np.array(np_series)
    np_series = np_series.reshape(1, np_series.shape[0], 1)
    return np_series


# In[ ]:


import pandas as pd
import numpy as np
# where can we see the values -- that are prevalent to the  training for the learning data_sets
# how is it possible to laod --> rest or all the elements in the datasets.
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
# load the data - 
scaler = MinMaxScaler(feature_range = (0, 1))
data = pd.read_csv("../working/data_matic_hourly.csv")
print(data.shape)
data["difference"] = data["high"] - data["low"]
data_high = data[["high"]].values
data_low = data[["low"]].values # high and low will be correlated with each other to run the program --> 
# some normalization for the data --. 
scaled_high = scaler.fit_transform(data_high)
all_data = []
target = []
all_data_low = []
target_low = []

for i in range(168, scaled_high.shape[0]):
    all_data.append(scaled_high[i-168:i, 0])
    target.append(scaled_high[i, 0])
    #all_data_low.append(scale)
all_data = np.array(all_data)
target = np.array(target)


cols = []

for i in range(all_data.shape[1]):
    temp =  "col"+ str(i)
    cols.append(temp)
print(len(cols))
#cols

# in order to see the correlation graph --> building the data frame is very critital
current_df = pd.DataFrame(data = all_data, columns = cols)
#current_df.corr()# there is significant correlation for the data --> 
current_df.head()
target = np.array(target.reshape(target.shape[0], 1))
all_data = all_data.reshape(all_data.shape[0], all_data.shape[1], 1)
all_data_train = all_data[:1500]
all_data_test = all_data[1500:]
train_target = target[:1500]
test_target = target[1500:]


print(all_data_train.shape, train_target.shape, all_data_test.shape, test_target.shape) # this is 

# values should be called with those --. 


# In[ ]:


# what kind of engineering works are great fit for loading the datasets -- all around here to maintain higher performance -- metrics --> 
# model creation for this is --> simple but --> prevalent to build it .
model = keras.models.Sequential()
model.add(keras.layers.LSTM(units = 100, return_sequences = True, input_shape = (all_data.shape[1], 1)))
model.add(keras.layers.LSTM(units = 100, return_sequences = True))
model.add(keras.layers.LSTM(units = 100, return_sequences = True))
model.add(keras.layers.LSTM(units = 100))
model.add(keras.layers.Dense(units = 1))
model.compile(optimizer = "adam", loss = "mean_squared_error")
model.summary()


# In[ ]:


epochs = 200# this is optimized hyper parameter
# for the test just train within in a single epochs --> 
# 7 days --> correlations -- trigger  the generality 
# batch_size -- smaller is  slower training through the details
history  = model.fit(all_data_train, train_target, epochs= epochs, batch_size=16, validation_data= (all_data_test, test_target)) 

history  = model.fit(all_data_train, train_target, epochs= 100, batch_size=100, validation_data= (all_data_test, test_target)) 
# 

# In[ ]:


# there should be some loads for the datasets -->  that is running on the current service --> 
# all the correlation -- within the data --> 
# can reset the model to run --> build on it. cdc -- 


# In[ ]:


print("Loss function representation ") # this is the scaled loss functions -- that is running -- 
import matplotlib.pyplot as plt
plt.plot(history.history["loss"])# can give the color coding here to build 
plt.plot(history.history["val_loss"])
plt.legend(["training_loss", "validation_loss"])
plt.show()


# In[ ]:


"""
test and training data_split approximate in size  -- similar loss values for good model.

"""


# In[ ]:


# not required to plot with this->
plt.plot(history.history["val_loss"]) #the validation loss -- will be updated when values  will be prevalent.
# there is signicant drop in the performance of the model --. when it runs on the server
# by showing the values the proper epochs would be 40
plt.show()
# by particular point -- required to have -- the scatter values --> 


# In[ ]:


#model.save("model_high_with_30_epochs.pb")
# there is some tendency for the graph that is missing very high defferentiation --> 


# In[ ]:


# when the program runs -- i will get the highest -- points --> 
# some components of the program will be suffled to run -


# In[ ]:


current = all_data_test[-1]
current = current.reshape(1, current.shape[0], current.shape[1])
value = model.predict(current)
predicted = scaler.inverse_transform(value)


ts = list(current.reshape(current.shape[1]))

ts.append(value[0][0]) # this is the non scaled value that needs to included in the this data --> 
ts = np.array(ts)
ts = ts.reshape(ts.shape[0], 1)
ts = scaler.inverse_transform(ts)
ts_high = ts.reshape(ts.shape[0])
print(ts_high.shape)
plt.plot(ts_high[-20:]) # some outliers -->  if it represents the 
plt.show()
predicted


# In[ ]:


current.shape


# In[ ]:


current = all_data_test[-1]
current = current.reshape(1, current.shape[0], current.shape[1])
def buildgraph(current, model):
    """
    This will get the particular time series then returns - the values -- useful for the next prediciton.
    current is test set to get the one stop data precition value -- np.array 1d
    
    """
    value = model.predict(current)
    predicted = scaler.inverse_transform(value)

    ts = list(current.reshape(current.shape[1]))

    ts.append(value[0][0]) # this is the non scaled value that needs to included in the this data --> 
    ts = np.array(ts)
    #ts = ts.reshape(ts.shape[0], 1)
    #ts = scaler.inverse_transform(ts)
    ts_high = ts
    ts_high = ts_high[1:]# comes into the the  same length of row features
    
    print(ts_high.shape)
    
    
    plt.plot(ts_high[-20:]) # some outliers -->  if it represents the 
    plt.legend(["prices_scaled"])
    plt.show()
    ts_high  = ts_high.reshape(1, ts_high.shape[0], 1)
    
    
    return ts_high, predicted


    
    
    
    #ts = ts[1:]
    #ts = list(ts)
    #ts.append(value)
    #ts = np.array(ts)
    #ts = ts.reshape(1, ts.shape[0], 1)
    #return ts
predicted_steps = []
# 20 data points will be calculated --> 
for i in range(20): # get 10 steps
    current, predicted_s = buildgraph(current, model)
    print(predicted_s)
    predicted_steps.append(predicted_s)
    # running for forecast of th 3 stmaage predictions --> 
    # can use the last real values.
    


# In[ ]:


predicted_steps # exposing --> 


# In[ ]:


import pandas as pd
import numpy as np
# where can we see the values -- that are prevalent to the  training for the learning data_sets
# how is it possible to laod --> rest or all the elements in the datasets.
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
# load the data - 
scaler = MinMaxScaler(feature_range = (0, 1))
data = pd.read_csv("../working/data_matic_hourly.csv")
print(data.shape)
data["difference"] = data["high"] - data["low"]
data_high = data[["low"]].values
data_low = data[["low"]].values # high and low will be correlated with each other to run the program --> 
# some normalization for the data --. 
scaled_high = scaler.fit_transform(data_high)
all_data = []
target = []
all_data_low = []
target_low = []

for i in range(168, scaled_high.shape[0]):
    all_data.append(scaled_high[i-168:i, 0])
    target.append(scaled_high[i, 0])
    #all_data_low.append(scale)
all_data = np.array(all_data)
target = np.array(target)



cols = []

#temporary data load and validation that can clearly be the higher performance loads -- withall the rest of the variables.


for i in range(all_data.shape[1]):
    temp =  "col"+ str(i)
    cols.append(temp)
print(len(cols))
#cols

# in order to see the correlation graph --> building the data frame is very critital
current_df = pd.DataFrame(data = all_data, columns = cols)
#current_df.corr()# there is significant correlation for the data --> 
current_df.head()
target = np.array(target.reshape(target.shape[0], 1))
all_data = all_data.reshape(all_data.shape[0], all_data.shape[1], 1)
all_data_train = all_data[:1500]
all_data_test = all_data[1500:]
train_target = target[:1500]
test_target = target[1500:]


print(all_data_train.shape, train_target.shape, all_data_test.shape, test_target.shape) # this is 


# In[ ]:



# model creation for this is --> simple but --> prevalent to build it .
model1 = keras.models.Sequential()
model1.add(keras.layers.LSTM(units = 100, return_sequences = True, input_shape = (all_data.shape[1], 1)))
model1.add(keras.layers.LSTM(units = 100, return_sequences = True))
model1.add(keras.layers.LSTM(units = 100, return_sequences = True))
model1.add(keras.layers.LSTM(units = 100))
model1.add(keras.layers.Dense(units = 1))
model1.compile(optimizer = "adam", loss = "mean_squared_error")
model1.summary() # there is the summary of the model that can be loaded into a several values.


# In[ ]:


# it is really important to make the current graph to the default --  
# significance correlation graph 
# same training is includedhere -- run --> 
epochs = 200 # setting this epochs into the  50 then -- loading  with the lower batch_size as with hyper parameter  optimization --> 
history = model1.fit(all_data_train, train_target, epochs= epochs, batch_size=16, validation_data= (all_data_test, test_target)) 


# In[ ]:


# with non loss function, it almost better to have smaller batch_size but not really best --> 
print("Loss function representation ") # this is the scaled loss functions -- that is running -- 
import matplotlib.pyplot as plt
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["training_loss", "validation_loss"])

plt.show()
plt.plot(history.history["val_loss"])
plt.legend(["validation_loss"])
plt.show()


# In[ ]:


current = all_data_test[-1]
current = current.reshape(1, current.shape[0], current.shape[1])
def buildgraph(current, model):
    """
    This will get the particular time series then returns - the values -- useful for the next prediciton.
    current is test set to get the one stop data precition value -- np.array 1d
    
    """
    value = model.predict(current)
    predicted = scaler.inverse_transform(value)# predicted is already scaled

    ts = list(current.reshape(current.shape[1]))

    ts.append(value[0][0]) # this is the non scaled value that needs to included in the this data --> 
    ts = np.array(ts)
    #ts = ts.reshape(ts.shape[0], 1)
    #ts = scaler.inverse_transform(ts)
    ts_high = ts
    ts_high = ts_high[1:]# comes into the the  same length of row features
    
    print(ts_high.shape)
    
    # this comes into the series that needs to run with following statements --> 
    plt.plot(ts_high[-20:]) # some outliers -->  if it represents the 
    plt.legend(["prices_scaled"])
    plt.show()
    ts_high  = ts_high.reshape(1, ts_high.shape[0], 1)
    
    
    return ts_high, predicted 
    
    #ts = ts[1:]
    #ts = list(ts)
    #ts.append(value)
    #ts = np.array(ts)
    #ts = ts.reshape(1, ts.shape[0], 1)
    #return ts
predicted_steps_low = []
# I will get the predictions for the next 20 time periods --> 
for i in range(20):
    current, predicted_s = buildgraph(current, model1)
    print(predicted_s)
    predicted_steps_low.append(predicted_s)
    # running for forecast of th 3 stage predictions --> 
    # can use the last real values.
    


# In[ ]:


highest_predicted = []
for i in predicted_steps:
    highest_predicted.append(i[0][0])# there

lowest_predicted = []
for i in predicted_steps_low:
    lowest_predicted.append(i[0][0])
    


# In[ ]:


# re-scale 


# In[ ]:


"""
current = all_data_test[-1]
current = current.reshape(1, current.shape[0], current.shape[1])
value = model1.predict(current)
predicted = scaler.inverse_transform(value)


ts = list(current.reshape(current.shape[1]))

ts.append(value[0][0]) # this is the non scaled value that needs to included in the this data --> 
ts = np.array(ts)
ts = ts[1:]
ts = ts.reshape(1, ts.shape[0], 1)
value = model1.predict(ts)
predicted1 = scaler.inverse_transform(value)
#ts = scaler.inverse_transform(value)
ts_high = ts.reshape(ts.shape[0])
# there are several functions that can run with higher performance metrics -->  
print(predicted, predicted1)
ts = list(current.reshape(current.shape[1]))
plt.plot(ts_high[-20:]) # some outliers -->  if it represents the 
plt.show()

 
"""


# In[ ]:


plt.plot(highest_predicted)
plt.plot(lowest_predicted)
plt.legend(["highest", "lowest"])
plt.title("hourly prices for next 10 hrs")
plt.xlabel("time points")
plt.ylabel("prices")


# In[ ]:


highest_predicted
# just let the model get the trained 


# In[ ]:


data[-20:][["high", "low"]].values[0][1]


# In[ ]:


import time
time.time()


# In[ ]:


model1.save(filepath = "../working/")


# In[ ]:


model.save(filepath  = "../working/")


# In[ ]:


model.summary()


# In[ ]:


model1.summary()


# In[ ]:


# there are crititcal points that runs -- with following elements -- that can be loaded --  with the rest of the elements .
import os


# In[ ]:


os.listdir() #there


# In[ ]:


current_model = keras.models.load_model("../working/")# there there are assets for the model loading --
"""
For following actions to take careful consideration on the dataflow -- that can""


# In[ ]:




