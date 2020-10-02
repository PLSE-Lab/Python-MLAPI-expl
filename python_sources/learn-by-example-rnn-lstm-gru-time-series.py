#!/usr/bin/env python
# coding: utf-8

# # Learn by example RNN/LSTM/GRU time series
# 
# I know I cannot predict stock prices based on historic data, but still the Recurring Neural network examples (RNN or LSTM or GRU, etc) to predict stock prices are appealing, who knows I might discover something:-)
# 
# Welcome to my second  notebook on Kaggle. I did record my notes so it might help others in their journey to understand Neural Networks by examples (in this case using Recurring Networks stock predictions.) After seeing many youtube video's and various courses on Neural Networks found the Kaggle Keras course and examples helping me a lot to move from powerpoint understanding to run my own Neural Networks using Keras. Many thanks to this community! The least I could do is to contribute back, hence this notebook.  
# 
# 

# # Table of Contents
#   
# 1. [Introduction](#sec1)
# 1. [Sinus wave proof of concept](#sec2)
#     1. [Results](#sec21)
# 1. [IBM stock prediction](#sec3)
#     1. [Everybody on Kaggle rich!](#sec31)
#     1. [you can't predict future stock prices on historic data](#sec32)
# 

# <a id='sec1'></a>
# # Introduction
# 
# Please watch this video about RNN/LSTM/GRU time series prediction, it gives you a good overview, it inspired me to reproduce the steps taken in this notebook:  
# https://www.youtube.com/watch?v=2np77NOdnwk  
# The code base:  
# https://github.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction  
# I would like to acknowledge this tutorial for providing ideas and code, learning by example:  
# https://www.kaggle.com/thebrownviking20/intro-to-recurrent-neural-networks-lstm-gru/notebook  
# If you are new to Neural Networks you might want to have a look at my first notebook:  
# https://www.kaggle.com/charel/learn-neural-networks-by-example-mnist-digits  
# 

# In[ ]:


import numpy as np # linear algebra
from numpy import newaxis
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

print ('import completed')


# In[ ]:


# Enter in how much steps we will enroll the network.
# RNN/LSTM/GRU can be taught patterns over times series as big as the number of times you enrol them, and no bigger (fundamental limitation). 
# So by design these networks are deep/long to catch recurrent patterns.
Enrol_window = 100

print ('enrol window set to',Enrol_window )


# In[ ]:


# Support functions
sc = MinMaxScaler(feature_range=(0,1))
def load_data(datasetname, column, seq_len, normalise_window):
    # A support function to help prepare datasets for an RNN/LSTM/GRU
    data = datasetname.loc[:,column]

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    
    if normalise_window:
        #result = sc.fit_transform(result)
        result = normalise_windows(result)

    result = np.array(result)

    #Last 10% is used for validation test, first 90% for training
    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

    return [x_train, y_train, x_test, y_test]

def normalise_windows(window_data):
    # A support function to normalize a dataset
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

def predict_sequence_full(model, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
    return predicted

def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of <prediction_len> steps before shifting prediction run forward by <prediction_len> steps
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs

def plot_results(predicted_data, true_data): 
    fig = plt.figure(facecolor='white') 
    ax = fig.add_subplot(111) 
    ax.plot(true_data, label='True Data') 
    plt.plot(predicted_data, label='Prediction') 
    plt.legend() 
    plt.show() 
    
def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()

print ('Support functions defined')


# <a id='sec2'></a>
# # Sinus wave proof of concept
# First let's run some tests on a plain sinus wave and see of the Neural Network can predict it right, kind of proof of concept

# In[ ]:


# Load the data
dataset = pd.read_csv('../input/sinwave/Sin Wave Data Generator.csv')
dataset["Wave"][:].plot(figsize=(16,4),legend=False)


# In[ ]:


# Prepare the dataset, note that all data foer the sinus wave is already normalized between 0 and 1
# A label is the thing we're predicting
# A feature is an input variable, in this case a stock price

feature_train, label_train, feature_test, label_test = load_data(dataset, 'Wave', Enrol_window, False)

print ('Datasets generated')


# In[ ]:


# The LSTM model I would like to test
# Note: replace LSTM with GRU or RNN if you want to try those

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(feature_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1, activation = "linear"))

model.compile(loss='mse', optimizer='adam')

print ('model compiled')

print (model.summary())


# In[ ]:


#Train the model
model.fit(feature_train, label_train, batch_size=512, epochs=10, validation_data = (feature_test, label_test))


# In[ ]:


#Let's use the model and predict the wave
predictions= predict_sequence_full(model, feature_test, Enrol_window)
plot_results(predictions,label_test)
  


# <a id='sec21'></a>
# ## Results
# Actually not a bad result because remember the models predicts 500 steps in the future and more-over after the enrol_window length (eg 100 setps) the predictions are being made on predictions, so eny error quickly multiplies ba magnitudes.  
# Having this confidence let's try to predicts some stock prices

# <a id='sec3'></a>
# # IBM stock prediction
# Could we predict stock prices with the neural network? Let's try it on some actual data. The daily stock prices of IBM stock are available for 2006-2017. Let's try to predict that last 10% of the data (approx the 2017 data) based on all the data before.

# In[ ]:


# Let's get the stock data
dataset = pd.read_csv('../input/stock-time-series-20050101-to-20171231/IBM_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])
dataset.head()


# In[ ]:


# Prepare the dataset, note that the stock price data will be normalized between 0 and 1
# A label is the thing we're predicting
# A feature is an input variable, in this case a stock price
# Selected 'Close' (stock pric at closing) attribute for prices. Let's see what it looks like

feature_train, label_train, feature_test, label_test = load_data(dataset, 'Close', Enrol_window, True)

dataset["Close"][:'2016'].plot(figsize=(16,4),legend=True)
dataset["Close"]['2017':].plot(figsize=(16,4),legend=True) # 10% is used for thraining data which is approx 2017 data
plt.legend(['Training set (First 90%, approx before 2017)','Test set (Last 10%, approax 2017 and beyond)'])
plt.title('IBM stock price')
plt.show()


# In[ ]:


# The same LSTM model I would like to test, lets see if the sinus prediction results can be matched
# Note: replace LSTM with GRU or RNN if you want to try those

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(feature_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1, activation = "linear"))

model.compile(loss='mse', optimizer='adam')

print ('model compiled')

print (model.summary())


# In[ ]:


#Train the model
model.fit(feature_train, label_train, batch_size=512, epochs=5, validation_data = (feature_test, label_test))


# In[ ]:


#Let's use the model and predict the stock
predicted_stock_price = model.predict(feature_test)
plot_results(predicted_stock_price,label_test)


# <a id='sec31'></a>
# ## Everybody on Kaggle rich! 
# This looks incredible correct, but "if it is to good to be true, it is probably not true".  
# Let's step back and actually see what we did. We created a testset of 100 (dependend on how you set the enrol_window) actual datapoints and ask to predict nr 101 (which is probably anhow close to nr 100). And we did so for each point in this graph. Hence the fantastic result, it wasn't that hard (Remember that you are looking at normalised data)
# 
# Like the sinewave example we need to predict a new point based on the actual last 100 points, the next point on 99 actual points and 1 prediction, the next point on 98 actuals and 2 predictions, and so forth.   
# Lets make some 50 predictions ahead in the future and do this every 50 times to get a bearing how the model predicts

# In[ ]:


predictions = predict_sequences_multiple(model, feature_test, Enrol_window, 50)
plot_results_multiple(predictions, label_test, 50)  


# <a id='sec32'></a>
# ##  You can't predict future stock prices on historic data
# Ouch, can't use that to put some real money in the stock market.  We basically knew already that you cant predict future stock prices on historic data. Pick for example the grey or purple line, it probably learned the stock went down last 100 sequence so it predicts it will go down, what would be a recognizable pattern to predict the trend will break and would go back up again after point 200.  So it is nog recognisable in the historic data, else the algorithem would have found it. Maybe with a richer data set with correleated stocks? Other (News?) items? Etc Anyhow, still a nice learning example which helped me to practice with LSTM (but could also picked GRU or RNN, fw simple code changes in the model)
