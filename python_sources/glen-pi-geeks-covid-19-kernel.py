#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #graphs, etc.
import math
import time
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import operator

# LSTM for international airline passengers problem with regression framing
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
plt.style.use('seaborn')
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#This project is for Covid-19 Forecasting
#Using a bubble graph to show spread of Covid-19
#We will start off with Bucks County and then move on


# In[ ]:


# import stuff

confirmed_totalcases = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
totaldeaths_reported = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')
recovered_cases = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')


# In[ ]:


# Death Rate - Machine Learning


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


deaths_reported = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')
death_report_cols = deaths_reported.keys()
deaths_reported[120:140]


# In[ ]:


dool_deaths_reported = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')
dool_death_report_cols = dool_deaths_reported.keys()


# In[ ]:


# Copy of dataframe
death_report_1 = deaths_reported

# Making a list of all rows to delete
droppings = death_report_1[death_report_1['Country/Region'] != 'US'].index

# Delete these row indexes from dataFrame
death_report_1.drop(droppings, inplace=True)
deaths_china = death_report_1
deaths_china_cols = deaths_china.keys()
# fulltable_us.info()
deaths_china.head()


# In[ ]:


#start
deaths = deaths_china.loc[:, deaths_china_cols[12]:deaths_china_cols[-1]]
deaths.head()


# In[ ]:


deaths_diff_sum = deaths.agg("sum", axis="rows")
print(deaths_diff_sum)

#get the diff
deaths_diff_sum_diff = deaths_diff_sum.diff()
deaths_diff_sum_diff.fillna(0, inplace=True)
print(deaths_diff_sum_diff)


# In[ ]:


# Copy of dataframe
dool_death_report_1 = dool_deaths_reported

# Making a list of all rows to delete
droppable = dool_death_report_1[dool_death_report_1['Country/Region'] != 'Italy'].index

# Delete these row indexes from dataFrame
dool_death_report_1.drop(droppable, inplace=True)
deaths_italy = dool_death_report_1
deaths_italy_cols = deaths_italy.keys()
# fulltable_us.info()
deaths_italy.head()


# In[ ]:


#start
dool_deaths = deaths_italy.loc[:, deaths_italy_cols[12]:deaths_italy_cols[-1]]
dool_deaths.head()


# In[ ]:


deaths_sum = dool_deaths.agg("sum", axis="rows")
print(deaths_sum)

#get the diff
training = deaths_sum.diff()
training.fillna(0, inplace=True)
print(training)


# In[ ]:


#plot the changes in death
plt.plot(deaths_diff_sum_diff)
plt.show()


# In[ ]:


dates = deaths.keys()


# In[ ]:


# Fu# Convert all dates and cases into the form of a numpy array

days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
# world_cases = np.array(world_cases).reshape(-1, 1)
death_rate = np.array(deaths_diff_sum_diff).reshape(-1, 1)
# total_recovered = np.array(total_recovered).reshape(-1, 1)


# In[ ]:


days_since_1_22


# In[ ]:


# Future forecasting for next 10 days

days_in_future = 0
future_forecast = np.array([i for i in range(len(dates) + days_in_future)]).reshape(-1, 1)
adjusted_dates = future_forecast[:-10]


# In[ ]:


# Import gang

from sklearn.model_selection import train_test_split
# X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22, death_rate, test_size=0.15, shuffle=False)


# In[ ]:


# Building the SVM model

"""kernel = ['poly', 'sigmoid', 'rbf']
c = [0.01, 0.1, 1, 10]
gamma = [0.01, 0.1, 1]
epsilon = [0.01, 0.1, 1]
shrinking = [True, False]
svm_grid = {'kernel': kernel, 'C': c, 'gamma' : gamma, 'epsilon': epsilon, 'shrinking' : shrinking}

svm = SVR()
svm_search = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
svm_search.fit(X_train_confirmed, y_train_confirmed)"""


# In[ ]:


# svm_search.best_params_


# In[ ]:


# svm_confirmed = svm_search.best_estimator_
# svm_pred = svm_confirmed.predict(future_forecast)


# In[ ]:


# svm_confirmed


# In[ ]:


# svm_pred


# In[ ]:


# check against testing data

"""svm_test_pred = svm_confirmed.predict(X_test_confirmed)
plt.plot(svm_test_pred)
plt.plot(y_test_confirmed)
print('MAE:', mean_absolute_error(svm_test_pred, y_test_confirmed))
print('MSE:', mean_squared_error(svm_test_pred, y_test_confirmed))"""


# In[ ]:


# total number of cases over time

"""plt.figure(figsize=(20, 12))
plt.plot(adjusted_dates, deaths_diff_sum_diff)
plt.title('Number of Coronavirus Cases Over Time', size=30)
plt.xlabel('Days since 1/22/2020', size=30)
plt.ylabel('Number of Cases', size=30)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()"""


# In[ ]:


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

# fix random seed for reproducibility
# numpy.random.seed(7)


# In[ ]:


# load the dataset
# dataframe = read_csv('airline-passengers.csv', usecols=[1], engine='python')
# dataset = dataframe.values
dataset = death_rate.astype('float64')
dataset


# In[ ]:


# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)


# In[ ]:


training_1 = np.array(training).reshape(-1, 1)
# split into train and test sets
train_size= int(len(training_1) * 0.67)
test_size = len(training_1) - train_size
train, test = training_1[0:train_size,:], training_1[train_size:len(training_1),:]


# In[ ]:


train.index


# In[ ]:


# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = 
testX, testY = 

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# In[ ]:


# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict


# In[ ]:


# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

