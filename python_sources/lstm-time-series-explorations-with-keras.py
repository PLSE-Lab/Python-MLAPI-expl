#!/usr/bin/env python
# coding: utf-8

# # LSTM Time Series Explorations with Keras
# 
# This is a very short exploration into applying LSTM techniques using the Keras library. Code and content is based on several cool blog posts and papers; see references section for more info.
# 
# There are further notes on LSTM theory and Keras in an accompanying Slideshare: https://www.slideshare.net/RalphSchlosser/lstm-tutorial
# 
# **FIXMEs**: 
# * Compare and contrast different parametrizations or architectures.
# * Consolidate helper functions, simplify.
# 
# ## Example 1: Airline Passenger Data
#  
# In this example we wish to make forcasts on a  time series of international airline passengers.
#  
# The time series data forcast can be modeled as a univariate regression-type problem, concretely let ${X_t}$ denote the number of airline passengers in month $t$. Then: 
#  
# $$
# X_t = f(X_{t-1}, \Theta)
# $$
#  
# which we aim to solve using the a simple LSTM neural network. 
# 
# Here $X_t$ is the number of passengers at time step $t$, $X_{t-1}$ denotes  number of passengers at the previous time step, and $\Theta$ refers to all the other model parameters, including LSTM hyperparameters.
# 
# *Note*: For better readability, in the code for this as well as the next example, the predicted new value at time step $t$ is written as `Y`. 
# 
# ### Loading and plotting the data

# In[ ]:


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Original data set retrieved from here:
# https://datamarket.com/data/set/22u3/international-airline-passengers-monthly-totals-in-thousands-jan-49-dec-60#!ds=22u3&display=line

data = pd.read_csv("../input/international-airline-passengers.csv", 
                      usecols = [1], 
                      engine = "python", 
                      skipfooter = 3)


# In[ ]:


# Print some data rows.
data.head()


# Here, we have a univariate data set which records the number of airline passengers for each month.
# 
# Let's now plot the time series of the data in order to get some ideas about underlying trends, seasonality etc.

# In[ ]:


# Create a time series plot.
plt.figure(figsize = (15, 5))
plt.plot(data, label = "Airline Passengers")
plt.xlabel("Months")
plt.ylabel("1000 International Airline Passengers")
plt.title("Monthly Total Airline Passengers 1949 - 1960")
plt.legend()
plt.show()


# In general we can observe a strong upwards trend in terms of numbers of passgengers with some seasonality component. The seasonality may be understood to conincide with holiday periods, but we'd need to have a closer look at the actual time periods to confirm this.
# 
# We could also consider de-trending the time series and applying further "cleaning" techniques, which would be a prerequisite e.g. in an *ARIMA* setting.
# 
# However, for simplicity reasons we will just proceed with the data as is.
# 
# The only transformations we'll be doing are:
# 
# * Scale data to the $(0, 1)$ interval for increased numerical stability.
# * Re-shape the data so we have one column as **response** (called $Y$ in the code) and another one as **predictor** variable (called $X in the code).

# ### Building the LSTM model
# 

# In[ ]:


# Let's load the required libs.
# We'll be using the Tensorflow backend (default).
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle


# #### Data preparation

# In[ ]:


# Get the raw data values from the pandas data frame.
data_raw = data.values.astype("float32")

# We apply the MinMax scaler from sklearn
# to normalize data in the (0, 1) interval.
scaler = MinMaxScaler(feature_range = (0, 1))
dataset = scaler.fit_transform(data_raw)

# Print a few values.
dataset[0:5]
dataset.shape


# #### Split into test / training data
# 
# As usual, the data gets split into training and test data so we can later assess how well the final model performs. 
# 
# Again, this could be much improved, e.g. using CV and more sophisticated steps to select the "best" model.

# In[ ]:


# Using 60% of data for training, 40% for validation.
TRAIN_SIZE = 0.60

train_size = int(len(dataset) * TRAIN_SIZE)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print("Number of entries (training set, test set): " + str((len(train), len(test))))


# #### Get data into shape to use in Keras

# In[ ]:


# FIXME: This helper function should be rewritten using numpy's shift function. See below.
def create_dataset(dataset, window_size = 1):
    data_X, data_Y = [], []
    for i in range(len(dataset) - window_size - 1):
        a = dataset[i:(i + window_size), 0]
        data_X.append(a)
        data_Y.append(dataset[i + window_size, 0])
    return(np.array(data_X), np.array(data_Y))


# In[ ]:


# Create test and training sets for one-step-ahead regression.
window_size = 1
train_X, train_Y = create_dataset(train, window_size)
test_X, test_Y = create_dataset(test, window_size)
print("Original training data shape:")
print(train_X.shape)


# Reshape the input data into appropriate form for Keras.
train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))
print("New training data shape:")
print(train_X.shape)


# #### Build simple LSTM model on training data
# 
# The LSTM architecture here consists of:
# 
# * One input layer.
# * One LSTM layer of 4 blocks.
# * One `Dense` layer to produce a single output.
# * Use MSE as loss function.
# 
# Many different architectures could be considered. But this is just a quick test, so we'll keep things nice and simple.

# In[ ]:


def fit_model_original(train_X, train_Y, window_size = 1):
    model = Sequential()
    model.add(LSTM(4, 
                   input_shape = (1, window_size)))
    model.add(Dense(1))
    model.compile(loss = "mean_squared_error", 
                  optimizer = "adam")
    model.fit(train_X, 
              train_Y, 
              epochs = 100, 
              batch_size = 1, 
              verbose = 2)
    return(model)

# Define the model.
def fit_model_new(train_X, train_Y, window_size = 1):
    model2 = Sequential()
    model2.add(LSTM(input_shape = (window_size, 1), 
               units = window_size, 
               return_sequences = True))
    model2.add(Dropout(0.5))
    model2.add(LSTM(256))
    model2.add(Dropout(0.5))
    model2.add(Dense(1))
    model2.add(Activation("linear"))
    model2.compile(loss = "mse", 
              optimizer = "adam")
    model2.summary()

    # Fit the first model.
    model2.fit(train_X, train_Y, epochs = 100, 
              batch_size = 1, 
              verbose = 2)
    return(model2)
#model1=fit_model_original(train_X, train_Y)

model2=fit_model_new(train_X, train_Y)


# In[ ]:





# ### Results
# #### Predictions and model evaluation
# 
# As can be seen below, already the simple model performs not too poorly. The advantage of using the RMSE is that it's in the same unit as the original data, i.e. 1.000 passengers / month.

# In[ ]:


def predict_and_score(model, X, Y):
    # Make predictions on the original scale of the data.
    pred_scaled =model.predict(X)
    pred = scaler.inverse_transform(pred_scaled)
    # Prepare Y data to also be on the original scale for interpretability.
    orig_data = scaler.inverse_transform([Y])
    # Calculate RMSE.
    score = math.sqrt(mean_squared_error(orig_data[0], pred[:, 0]))
    return(score, pred, pred_scaled)

rmse_train, train_predict, train_predict_scaled = predict_and_score(model2, train_X, train_Y)
rmse_test, test_predict, test_predict_scaled = predict_and_score(model2, test_X, test_Y)

print("Training data score: %.2f RMSE" % rmse_train)
print("Test data score: %.2f RMSE" % rmse_test)

test_predict.size


# In[ ]:


#print(test_X.shape)
#print(test_X[0:1,:,:].shape)
#print(test_X[0:1,:,:])
X_single = test_X[0:1,:,:]
#print(test_Y.shape)
#print(test_Y[0:1].shape)
#print(test_Y[0:1])

# create empty array
from numpy import empty
test_predict_at_a_time = empty([test_X.size,1])
 
print("initial X: ", X_single)
for i in range((test_X.size)):
    Y_single = test_Y[i:i+1]
    rmse_test, predict, predict_scaled = predict_and_score(model2, X_single, Y_single)
    test_predict_at_a_time[i]= predict
    print("Test data score: %.2f RMSE" % rmse_test)
    print("predicted: ", predict[0])
    X_single = predict_scaled.copy() 
    X_single=np.reshape(X_single[0], (1, 1, 1))
    print("given X: ", X_single)
test_predict_at_a_time[-3:]


# #### Plotting original data, predictions, and forecast
# 
# With a plot we can compare the predicted vs. actual passenger figures.

# In[ ]:


# Start with training predictions.
train_predict_plot = np.empty_like(dataset)
train_predict_plot[:, :] = np.nan
train_predict_plot[window_size:len(train_predict) + window_size, :] = train_predict

# Add test predictions.
test_predict_plot = np.empty_like(dataset)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (window_size * 2) + 1:len(dataset) - 1, :] = test_predict


# Add test predictions.
test_predict_at_a_time_plot = np.empty_like(dataset)
test_predict_at_a_time_plot[:, :] = np.nan
test_predict_at_a_time_plot[len(train_predict) + (window_size * 2) + 1:len(dataset) - 1, :] = test_predict_at_a_time

# Create the plot.
plt.figure(figsize = (15, 5))
plt.plot(scaler.inverse_transform(dataset), label = "True value")
plt.plot(train_predict_plot, label = "Training set prediction")
plt.plot(test_predict_plot, label = "Test set prediction")
plt.plot(test_predict_at_a_time_plot, label = "Test set prediction at a time")
plt.xlabel("Months")
plt.ylabel("1000 International Airline Passengers")
plt.title("Comparison true vs. predicted training / test")
plt.legend()
plt.show()


# ### Next steps and things to explore
# 
# * Work with de-trended, stationary time series. Does it improve performance?
# * Different window size (multiple regression). See Example 2.
# * LSTM architecture, i.e. more layers, neurons etc.
# * Impact of various hyperparameters in LSTM network on prediction accuracy.
# * Model selection steps to find the "best" model.

# ## Example 2: Sinus wave data
# 
# For the second example, we'll generate synthetic data from a sinus curve, i.e.: $y = sin(x)$.
# 
# Unlike in the example above, now we'll build a *multiple regression* model where we treat a range of input values at previous time steps as inputs for predicting the output value at the next time step. 
# 
# The number of previous time steps is called the *window size*. In the below we'll be using a window size of $50$, i.e.: 
# 
# $$
# X_t = f(X_{t-1}, X_{t-2}, ..., X_{t-50}, \Theta)
# $$
# 
# ### Generating and plotting the data

# In[ ]:


SAMPLES = 5000
PERIOD = 50
x = np.linspace(-PERIOD * np.pi, PERIOD * np.pi, SAMPLES)
series = pd.DataFrame(np.sin(x))

plt.figure(figsize = (15, 5))
plt.plot(series.values[:PERIOD])
plt.xlabel("x")
plt.ylabel("y")
plt.title("y = sin(x), first %d samples" % PERIOD)
plt.show()


# ### Building the LSTM model
# 
# #### Data preparation
# 
# First, we'll demonstrate the sliding window principle using a window size of 1; subsequently we'll move on to window size 50.

# In[ ]:


# Normalize data on the (-1, 1) interval.
scaler = MinMaxScaler(feature_range = (-1, 1))
scaled = scaler.fit_transform(series.values)

# Convert to data frame.
series = pd.DataFrame(scaled)

# Helper function to create a windowed data set.
# FIXME: Copying & overwriting is flawed!
def create_window(data, window_size = 1):    
    data_s = data.copy()
    for i in range(window_size):
        data = pd.concat([data, data_s.shift(-(i + 1))], 
                            axis = 1)
        
    data.dropna(axis=0, inplace=True)
    return(data)

# FIXME: We'll use this only for demonstration purposes.
series_backup = series.copy()
t = create_window(series_backup, 1)
t.head()


# In[ ]:


window_size = 50
series = create_window(series, window_size)
print("Shape of input data:")
print(series.shape)


# #### Split into training / test set

# In[ ]:


# Using 80% of data for training, 20% for validation.
# FIXME: Need to align with example 1.
TRAIN_SIZE = 0.80

nrow = round(TRAIN_SIZE * series.shape[0])

train = series.iloc[:nrow, :]
test = series.iloc[nrow:, :]

# Shuffle training data.
train = shuffle(train)

train_X = train.iloc[:, :-1]
test_X = test.iloc[:, :-1]

train_Y = train.iloc[:, -1]
test_Y = test.iloc[:, -1]

print("Training set shape for X (inputs):")
print(train_X.shape)
print("Training set shape for Y (output):")
print(train_Y.shape)


# #### Get data into shape to use in Keras

# In[ ]:


train_X = np.reshape(train_X.values, (train_X.shape[0], train_X.shape[1], 1))
test_X = np.reshape(test_X.values, (test_X.shape[0], test_X.shape[1], 1))

print(train_X.shape)
print(test_X.shape)


# #### Build LSTM model on training data
# 
# The model architecture used here is slightly more complex. Its elements are:
# 
# * LSTM input layer with 50 units.
# * `Dropout` layer to prevent overfitting (see: http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf).
# * A second LSTM layer with 256 units.
# * A further `Dropout` layer.
# * A `Dense` layer to produce a single output.
# * Use MSE as loss function.

# In[ ]:


# Define the model.
model2 = Sequential()
model2.add(LSTM(input_shape = (window_size, 1), 
               units = window_size, 
               return_sequences = True))
model2.add(Dropout(0.5))
model2.add(LSTM(256))
model2.add(Dropout(0.5))
model2.add(Dense(1))
model2.add(Activation("linear"))
model2.compile(loss = "mse", 
              optimizer = "adam")
model2.summary()


# ### Results
# #### Predictions and model evaluation

# In[ ]:


# Fit the model.
model2.fit(train_X, 
          train_Y, 
          batch_size = 512,
          epochs = 3,
          validation_split = 0.1)


# In[ ]:


# Predict on test data.
pred_test = model2.predict(test_X)

# Apply inverse transformation to get back true values.
test_y_actual = scaler.inverse_transform(test_Y.values.reshape(test_Y.shape[0], 1))

print("MSE for predicted test set: %2f" % mean_squared_error(test_y_actual, pred_test))

plt.figure(figsize = (15, 5))
plt.plot(test_y_actual, label="True value")
plt.plot(pred_test, label="Predicted value")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.title("Comparison true vs. predicted test set")
plt.legend()
plt.show()


# ### Next steps and things to explore
# 
# * Should clean up first before doing anything else! ;)
# 

# ## References and links
# 
# * Example 1 source code adapted from here: https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
# * Another great post: https://towardsdatascience.com/using-lstms-to-forecast-time-series-4ab688386b1f
# * Example 2 code adapted from the above: https://github.com/kmsravindra/ML-AI-experiments/tree/master/AI/LSTM-time_series
# * Good paper comparing different time series modeling techniques, including LSTM: https://arxiv.org/pdf/1705.09137.pdf
# * Another excellent paper: https://arxiv.org/pdf/1705.05690.pdf
# * Brilliant LSTM course by Nando de Freitas: https://www.youtube.com/watch?v=56TYLaQN4N8
# 
