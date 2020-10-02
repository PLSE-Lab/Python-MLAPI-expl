#!/usr/bin/env python
# coding: utf-8

# 1. Introduction
# 
#     1.1. Background
#     
#     1.2. What is  Time-Series anyway?
#     
#     1.3. Gathering Data
#     
#     
# 2. Data Exploration
# 
#     2.1. Plotting the Time-series
#     
# 
# 3. Finding Regressor Inputs
# 
#     3.1 Autocorrelation plot
#     
#     3.2. Creation of Dataset for training
#     
# 
# 4. Making a Neural Network
# 
# 
# 5. Splitting the data
# 
# 
# 6. Scaling the data
# 
# 
# 7. Training and Validation
# 
# 
# 8. Making Predictions on the test set
# 
#     8.1 Checking the r2 score
#     
#     8.2 Plotting the predictions

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt # data visualization

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# **1. Introduction**

# **1.1. Background**
# 
# With my current project, that intrduced me to Machine Learning, I have had the wonderful opportunity to develop a real-time Electrical Load forecasting system for my state. To help me with that, I have been cleaning my hands on many other Machine Learning endeavors, mostly involving time-series forecasting.
# 

# **1.2. What is a Time-Series anyway?**
#                                             
# A time-series is very simple information : Data measured against points in time. The resolution can be minutes like in weather, it can be in days like in the closing market value of a stock, it can be in months like in airline monthly ridership, it can be in years like in annual GDP of a country. 
# 
# Time-series forecasting is exactly what it sounds like - predicting future values in time. Developing a  forecasting model depends on the forecasting horizon. Very short-term forcasting would typically involve predicting the value at the next minute or hour, short-term forecasting has a horizon of days or weeks, long-term forecasting is done for months or a year or two, and we also have very-long term forecasting that could easily have a horizon of up to a few decades.
# 
# Here, we will build a simple Deep Learning based very-short-term Time-series forcaster, that will predict the stock prices of Bitcoin. 
# 
# I will also try to write code that is as generic as possible, so that it can be easily replicated to do time-series forecasting for a wide range of datasets and scenarios.

# **1.3. Gathering data**
# 
# Many of us are aware how dreadful the whole process of data collection and data comp
# can get. With stock prices however, it's a real piece of cake. One can download stock data for any company for any period in history, with any resoltion, from 'yahoo finance'. All you need to do is import the 'yahoo finance' module and check out the stock market code for the company that you are interested in.
# 
# I have chosen Bitcoin since from what I've heard, it's value is quite volatile, and I am a firm believer that a prediction model should be put through the most challenging task as possible. 
# 
# Make sure that Internet is on in the bottom right part of your noteboon where you can see the settings. We have to install the 'yfinance' package using pip

# In[ ]:


# pip install yfinance
# import yfinance
# df = yf.download('BTC-USD','2017-01-02','2019-11-16')


# I can't install 'yfinance' since that I haven't verified my phone on Kaggle yet. 
# 
# So, I have uploaded the dataset (downloaded from the yahoo finance website
# [Bitcoin USD historical data](http://https://finance.yahoo.com/quote/BTC-USD/history?period1=1483209000&period2=1573756200&interval=1d&filter=history&frequency=1d)).

# In[ ]:


df = pd.read_csv(r'/kaggle/input/bitcoin-usd-stock-prices/bitcoin.csv')


# **2. Data Exploration**
# 
# Let's have a quick look first

# In[ ]:


df.head()


# So, we have a datetime column, which is nice. 
# 
# Then the opening value at the start of every trading day, the closing values at whatever time traders must have call it a day, the highest and lowest values during each, and two other columns which I'm not so sure about, so I won't bother.
# 
# What we are interested in is just the closing value.

# In[ ]:


df_close = pd.DataFrame(df['Close'])


# In[ ]:


df_close.index = pd.to_datetime(df['Date'])


# It's better to have the DataFrame index in datetime format (pd.datetime) since it makes plotting, and slicing easier

# In[ ]:


df_close.index


# In[ ]:


df_close.head()


# In[ ]:


df_close.describe()


# So, we have data that spans over some 1050 days. The maximum Bitcoin hit during this period was some 19K USD, and it also fell in the seven hundred range. Boy, Bitcoin has really seen it all.
# 
# Let's plot it to get a better look.

# **2.1. Plotting the Time-series**

# In[ ]:


from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

plt.figure(figsize=(8, 6))
plt.plot(df_close, color='g')
plt.title('Bitcoin Closing Price', weight='bold', fontsize=16)
plt.xlabel('Time', weight='bold', fontsize=14)
plt.ylabel('USD ($)', weight='bold', fontsize=14)
plt.xticks(weight='bold', fontsize=12, rotation=45)
plt.yticks(weight='bold', fontsize=12)
plt.grid(color = 'y', linewidth = 0.5)


# **3. Finding Regressor Inputs**
# 
# There are many time-series forcasting techinques. First, there are the conventional statistical techniques like ARIMA, and exponential smoothing. Then there are other traditional approaches like Regression. Neural Networks have been popular in forecasting problems because of their ability to make predictions on non-linear complex data. Conventional statistical techniques don't work well with highly non-linear data. Multi-dimensional linear Regression has shown good results too on complex non-linear data (don't let the "linear" in it confuse you, it refers to the linear combination of input variables, the input variables themselves however can be expressed as higher degree polynomaials in the function that we are trying to model)
# 
# But we will do our predciting using a Neural Network. There are two basic paths to implement a model.
# 
# Some implement by considering the historical values of the very series they are trying to predict as the input regressors (variables). This approach assumes that the future values can be expressed as a fuction of the past values. The two things to determine are - which past values to use (how far back into the past we need to look), and the parameters of the function itself
# 
# Many times, we also include input variables that are of very different nature than the time-series under observation. For example, in Electrical load forecating the weather time series (temperature) is used as one of the input variables. It makes sense since in most regions of our planet, temperature does affect the electrical power consumption. A drop in the temperate regions would have people using more of their heaters, and in sub-tropical or tropical regions, a rise in temperatue typically drives the air-conditioning load.
# 
# Okay. So, first see how far into the past we need to look.

# **3.1. Autocorrelation plot**

# In[ ]:


from statsmodels.tsa import stattools

acf_djia, confint_djia, qstat_djia, pvalues_djia = stattools.acf(df_close,
                                                             unbiased=True,
                                                             nlags=50,
                                                             qstat=True,
                                                             fft=True,
                                                             alpha = 0.05)

plt.figure(figsize=(7, 5))
plt.plot(pd.Series(acf_djia), color='r', linewidth=2)
plt.title('Autocorrelation of Bitcoin Closing Price', weight='bold', fontsize=16)
plt.xlabel('Lag', weight='bold', fontsize=14)
plt.ylabel('Value', weight='bold', fontsize=14)
plt.xticks(weight='bold', fontsize=12, rotation=45)
plt.yticks(weight='bold', fontsize=12)
plt.grid(color = 'y', linewidth = 0.5)


# To see which past variables can be used as input in our model, we check the auto-correlation of the time-series. Most of you must be familiar with correlation. It's a way to measure the strength of association between any two variables. As the correlation value approaches 1, it indicates high postive association. Correlation near 0 indicates almost no association at all, and a values near -1 would indicate strong negative correlation.
# 
# In autocorrelation, we take the time-series value at current instant as one variable and one of the values from a time-instant in the past as the other variable, and find the correlation between them.
# 
# Here, I have used 'stattools.acf' (acf for autocorrelation fucntion) to plot the autocorrelation with 50 past time-instants (lags). It's from the 'statsmodels' library. It's a great tool for doing a wide range of statistical analysis. Don't worry if you don't get this particular fuction in the first look. All you need to know is that is this function returns four outputs, out of which the first one which I have assigned the name 'acf_djia' is the list containing autocorrelation values of the number of lags ('nlags') specified.
# 
# Choosing a lot of input attributes, specially if they don't have a high enough correlation with the target attribute, could actually harm us. I will select all the lags (past values) that have at least a correlation of 0.9 with the present value. From the plot we can see that number is 15.

# TLDR : What this means is that we will use the closing price of past 15 days to predict the closing price of any particular day. 
# 
# So, if you have the closing prices of any particular stock up until today, you can use this model to predict the closing price tomorrow.
# 
# Well, you only need today's closing price and the closing price of past fourteen days to predict tomorrow's closing price.
# 
# We still do need all the hsitorical data that we have to train, and test out model.

# **3.2. Creation of dataset for training**
# 
# Right now we only have one column - the actual closing price of Bitcoin (apart from the datetime index of course). To train our neural network, we have decided to use the past 15 values as inputs with the value at any corresponding time instant. This means we will have now an additional 15 columns where each row will have the closing price at any day and the corresponding prices of the past fifteen days.
# 
# I have created a function to create those columns. 

# In[ ]:


def create_regressor_attributes(df, attribute, list_of_prev_t_instants) :
    
    """
    Ensure that the index is of datetime type
    Creates features with previous time instant values
    """
        
    list_of_prev_t_instants.sort()
    start = list_of_prev_t_instants[-1] 
    end = len(df)
    df['datetime'] = df.index
    df.reset_index(drop=True)

    df_copy = df[start:end]
    df_copy.reset_index(inplace=True, drop=True)

    for attribute in attribute :
            foobar = pd.DataFrame()

            for prev_t in list_of_prev_t_instants :
                new_col = pd.DataFrame(df[attribute].iloc[(start - prev_t) : (end - prev_t)])
                new_col.reset_index(drop=True, inplace=True)
                new_col.rename(columns={attribute : '{}_(t-{})'.format(attribute, prev_t)}, inplace=True)
                foobar = pd.concat([foobar, new_col], sort=False, axis=1)

            df_copy = pd.concat([df_copy, foobar], sort=False, axis=1)
            
    df_copy.set_index(['datetime'], drop=True, inplace=True)
    return df_copy


# In[ ]:


list_of_attributes = ['Close']

list_of_prev_t_instants = []
for i in range(1,16):
    list_of_prev_t_instants.append(i)

list_of_prev_t_instants


# Here, we only have one time-series that is the closing price everyday. Using the function that I've defined, we can create regressor attributes for muliple columns in a single dataframe. For example, say we want to model current electrical load using not only past load values but also past temperature values.
# 
# Also, we can specify which past values to use exaclty in the form of a list. It's heplful in cases where the past regressors can be different than simply the previous fifteen values. In our case it is so because the auto-correlation line was an almost straight line with a negative line.

# In[ ]:


df_new = create_regressor_attributes(df_close, list_of_attributes, list_of_prev_t_instants)
df_new.head()


# In[ ]:


df_new.shape


# It's a very messed up looking function, I know. 
# 
# As you can see - this new dataset has the original time-series ('Close') and other 15 columns that are the past values taken as regressor inputs. 
# 
# Also, it starts from the 16th Jan, 2017, but the original time-series started from 1st Jan, 2017. I have done this to avoid NaN values that would appear in the new added columns for obvious reasons.
# 
# The best way I can explain this function is to imagine our original time-series as a cake that had 1049 slices. I can take certain number of slices from this cake in the same order and make a new cake from it using some magic, so that my original cake with its original size will always remain there.
# 
# First I took slices from the 15th slice to the last slice from my original cake and added it to my new platter ('df_copy'). This new cake has 1034 slices (made up from 15th to the last slice of my original cake).
# 
# I again take 1034 slices from my new cake and add it to the platter (append new column to the dataframe 'df_copy), but this one consists of 14th slice to the second last slice of my original cake.
# 
# Then I take 13th slice to the third last slice, and so on...
# 
# So, df_copy contains 16 new cakes, all having 1034 slices, but with all the desired cakes (columns)

# Okay, phew..
# 
# We have out dataset. Let's now build our neural network.

# **4. Making the neural Network**

# I will be training a simple Multi-layer Perceptron that has an input layer with 15 nodes (accounting for each of the 15 past regressor inputs).
# 
# It will have 2 hidden layers (yeah, 2 and that's what makes it "DEEP", nothing more, nothing less). I'll consider 60 nodes in each.
# 
# Why 60? I tried a lot of trials and combinations, and I was satisfied with the results of this one. 
# 
# And hyperparameter tuning (adjusting the model parameters to find the optimal combination) won't be a part of this exercise. 
# 

# In[ ]:


from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint


# In[ ]:


input_layer = Input(shape=(15), dtype='float32')
dense1 = Dense(60, activation='linear')(input_layer)
dense2 = Dense(60, activation='linear')(dense1)
dropout_layer = Dropout(0.2)(dense2)
output_layer = Dense(1, activation='linear')(dropout_layer)


# In[ ]:


model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()


# In[ ]:


from tensorflow.keras.utils import plot_model
plot_model(model)


# Those familiar with Neural networks and Keras can skip this part. 
# 
# Those very new to Machine Learning can have a glance to absorb enough to move forward. 
# 
# An input layer only has nodes. It's only job is take in the data and pass it on the hidden layer.
# 
# A dense layer has the links (edges) connected to it, with weights and everything. It also has an activation function.
# 
# The data from each input node gets multiplied with the weights of the links via which those input nodes are connected to the first hidden layer. Each node of the hidden layer then combines all the inputs to it (one from each input node) and passes through an activation fuction.
# 
# Similar process repeats for the second hidden layer.
# 
# A dropout layer is added before the output layer. It drops a certain percentage (20% in our case) of the links randomly, that are connected to the output node. It's a good practice and generally helps in curbing overfitting.
# 
# We simply define each layer and the preceding layer it's connected to, and then package them all together using the function 'Model'
# 
# The 15 new columns dataframe (df_new) will be passed through this network, one row at a time. The network will spit out an answer which will be compared with the value in the first column. This first prediction will be based on weights that mare set randomly. The error in the prediction will be used to change the weights in such a way that the prediction is better.
# 
# This will happen for all the 1034 rows. We can adjust the weights after passing through multiple samples (rows) in what we call a "batch". When all the rows have passed in this way, we say one "epoch" is over. We go through many epochs.
# 
# But how many epochs do we have to go through, you ask? You already might have guessed. When we get the feel that out neural network model has captured the patterns in the data and is in a position to make predictions. How do we ensure that?
# 
# What we do is hold out a few samples of data as a validation set. We don't use them for training (i.e for updating the weights). We only run the validation set after every epoch, and check the error. We also keep a track of the training error but it's not as important as validation error since the training error is what adjusts the weights. So, in a sense the training data is what the network soaks in. The validation data is something it doesn't soak in. As we keep checking the errors, only when the validation error has come down to near the training error, we stop further training.
# 
# There are many ways to calcualte error. Here, we have used "mean_squared_error", and also used the "adam" optimizer algorith that is responsible for updating the weights (see "model.compile" part).
# 
# We can also hold out an additional test set just to see how out new baby performs on more unseen data.

# **5. Spliting the data**

# We will separate out 5% of the samples (rows) in a random fashion for later testing purposes. 
# 
# The remaining 95 % is again split randomly. Some 5% of it is used as validation set and the remaining as training set.
# 
# All this splitting happens row wise.
# 
# Note that we also need to split it column wise as well. The first column are our actual values (target). These aren't fed to the neural network. The remaining 15 columns (input regressors) are fed to it.
# 
# So, the slicing of data happens both row wise and column wise. The sliced dataframes having only the regressor input columns typically have a "X" in the names we assign them, and the sliced dataframes (or series) with only the actual values (also called target values, as in being "targets" for the neural netwroks to accurately predict while training) have a "y" in the names we assign them.
# 
# Note that since this is time-series data, splitting the dataset randomly isn't a sensible thought. So, for time-series data, we will simply split in an orderly fashion.

# In[ ]:


test_set_size = 0.05
valid_set_size= 0.05

df_copy = df_new.reset_index(drop=True)

df_test = df_copy.iloc[ int(np.floor(len(df_copy)*(1-test_set_size))) : ]
df_train_plus_valid = df_copy.iloc[ : int(np.floor(len(df_copy)*(1-test_set_size))) ]

df_train = df_train_plus_valid.iloc[ : int(np.floor(len(df_train_plus_valid)*(1-valid_set_size))) ]
df_valid = df_train_plus_valid.iloc[ int(np.floor(len(df_train_plus_valid)*(1-valid_set_size))) : ]


X_train, y_train = df_train.iloc[:, 1:], df_train.iloc[:, 0]
X_valid, y_valid = df_valid.iloc[:, 1:], df_valid.iloc[:, 0]
X_test, y_test = df_test.iloc[:, 1:], df_test.iloc[:, 0]

print('Shape of training inputs, training target:', X_train.shape, y_train.shape)
print('Shape of validation inputs, validation target:', X_valid.shape, y_valid.shape)
print('Shape of test inputs, test target:', X_test.shape, y_test.shape)


# **6. Scaling the data**

# We will normalize our data in the range in the range (0.01, 0.99) before feeding it to the the neural network.

# Note that MinMaxScaler() function takes in dataframes or series (can also take in arrays), but it always returns n-dimensional arrays.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

Target_scaler = MinMaxScaler(feature_range=(0.01, 0.99))
Feature_scaler = MinMaxScaler(feature_range=(0.01, 0.99))

X_train_scaled = Feature_scaler.fit_transform(np.array(X_train))
X_valid_scaled = Feature_scaler.fit_transform(np.array(X_valid))
X_test_scaled = Feature_scaler.fit_transform(np.array(X_test))

y_train_scaled = Target_scaler.fit_transform(np.array(y_train).reshape(-1,1))
y_valid_scaled = Target_scaler.fit_transform(np.array(y_valid).reshape(-1,1))
y_test_scaled = Target_scaler.fit_transform(np.array(y_test).reshape(-1,1))


# So, the data is ready. The neural network is ready. Let's go.

# **7. Training and Validation**

# In[ ]:


model.fit(x=X_train_scaled, y=y_train_scaled, batch_size=5, epochs=30, verbose=1, validation_data=(X_valid_scaled, y_valid_scaled), shuffle=True)


# In case the validation loss had remained significantly more than the training loss at the end of training, but was continously coming down in the last few epochs, it would have been an indication that we need to run our model for more epochs, as the model needed more training.
# 
# If the validation loss had remained significantly more than the training loss at the end of training, and also become more or less static, then it would have been an indication of "overfitting", that is our model works far too well on the training data, it has perfectly captured it, including the noise, but it would work poorly on unseen data. In such a case, it's advised to increase the value in the dropout layer, reduce number of epcohs, increase the batch size, reduce the number of hidden layers, reduce the number of nodes in the hidden layers.
# 
# Our Validation loss doesn't seem to have changed much, specially when compared to the training loss. We can attribute that to the small number of training examples that we have, in the context of the model that we have used.
# 
# LSTMs and CNNs may be able to give better results than a conventional NN on the same data. That's for another time. We only want to get our first hands on experience for now.

# **8. Making predictions on the test set**

# In[ ]:


y_pred = model.predict(X_test_scaled)


# Recall that all our inputs and targets were scaled down in the range (0, 1). So, the predictions also lie in that range. We need to scale them back in the other direction

# In[ ]:


y_pred_rescaled = Target_scaler.inverse_transform(y_pred)


# **8.1. Checking the r2 score **

# One of the ways to measure the performance of our model on the test data is to compare the error of its predictions with respect to the true values. That could be mean squared error, or mean average error etc.
# 
# We could also use use something called as r_squared (or r2) score. Just remember that like Regression, it measures the squared mean distance between true values and values lying the predictor hyperplane (our predicted values), and spits out a score between 0 and 1. More the r2 score closer to one, better the predictions of your model

# In[ ]:


from sklearn.metrics import r2_score
y_test_rescaled =  Target_scaler.inverse_transform(y_test_scaled)
score = r2_score(y_test_rescaled, y_pred_rescaled)
print('R-squared score for the test set:', round(score,4))


# **8.2 Plotting the predictions **

# We can add the datatime index so that it automatically takes care of the x-axis tick labels.

# In[ ]:


y_actual = pd.DataFrame(y_test_rescaled, columns=['Actual Close Price'])

y_hat = pd.DataFrame(y_pred_rescaled, columns=['Predicted Close Price'])


# In[ ]:


plt.figure(figsize=(11, 6))
plt.plot(y_actual, linestyle='solid', color='r')
plt.plot(y_hat, linestyle='dashed', color='b')

plt.legend(['Actual','Predicted'], loc='best', prop={'size': 14})
plt.title('Bitcoin Stock Closing Prices', weight='bold', fontsize=16)
plt.ylabel('USD ($)', weight='bold', fontsize=14)
plt.xlabel('Test Set Day no.', weight='bold', fontsize=14)
plt.xticks(weight='bold', fontsize=12, rotation=45)
plt.yticks(weight='bold', fontsize=12)
plt.grid(color = 'y', linewidth='0.5')
plt.show()


# I guess the predictions are pretty reliable, at least for a forecasting horizon of one day. Not that it has brought me any closer to owning a bitcoin. Well, Bitcoin's not doing that great either. Plenty of better stock options out there.
# 
# I hope this kernel was helpful to you in one or many ways. I believe this code can be easily reused with very few modifications on any time-series forecasting problem, specially those with short term horizons.
