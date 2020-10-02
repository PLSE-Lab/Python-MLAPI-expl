#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from pandas.plotting import scatter_matrix

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from keras.models import Sequential
from keras.layers import Dense   
from keras import optimizers

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from pandas import DataFrame
from pandas import concat
from keras.models import load_model
from keras import optimizers
from matplotlib import pyplot
from math import sqrt
from keras import optimizers

# Any results you write to the current directory are saved as output.


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ### Let us read the US Counties dataset into a dataframe ###

# In[ ]:


np.random.seed(7)

# load the dataset
dataframe = pandas.read_csv('/kaggle/input/us-counties-covid-19-dataset/us-counties.csv')


# In[ ]:


dataframe.head()


# ### We sort the values by date ascending order ###

# In[ ]:


dataframe['pd_date'] = pd.to_datetime(dataframe.date)
dataframe=dataframe.sort_values(by='pd_date',ascending=True)


# ### We see we have data from 21-Jan to 11-Apr ###

# In[ ]:


print(dataframe['pd_date'].max())


# In[ ]:


print(dataframe['pd_date'].min())


# In[ ]:


dataframe.county.value_counts()[:20]


# ### Let us investigate the values for one County (Washington) within a state ('Oregon') ###

# In[ ]:


dataframe_wton = dataframe[(dataframe.county=='Washington') & (dataframe.state=='Oregon')]


# In[ ]:


len(dataframe_wton)


# In[ ]:


# Filter out only the cases and deaths values
#dataframe_wton.index=dataframe_wton['pd_date']
dataframe_wton = dataframe_wton.iloc[:,4:6]


# In[ ]:


dataframe_wton.head(20)


# In[ ]:


dataset_cases = dataframe_wton.values[:,0:1]
dataset_cases = dataset_cases.astype('float32')
dataset_deaths = dataframe_wton.values[:,1:2]
dataset_deaths = dataset_deaths.astype('float32')


# In[ ]:


plt.title("Number of COVID 19 cases by day for Washington County")
plt.plot(dataset_cases)
plt.show()


# In[ ]:


plt.title("Number of COVID 19 deaths by day for Washington County")
plt.plot(dataset_deaths)
plt.show()


# In[ ]:


dataset = dataset_cases


# # We split the train and test data. We have 44 values, worth of few months of data. We take 30 values as test data, and remaining as train data. Since this is time series data, we take the first 30 values as train data and next 14 values as test data

# In[ ]:


train_size = int(len(dataset)) - 14
#test_size = 14
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))


# ### **Approach 1: We use lookback period of 5, and create Supervised Training Data** for applying Deep Learning

# ### The next part is the most important step in creating the train data for a Supervised Learning problem. We shift the data with a lookback of 5 time steps and use this data to predict the value of the Call volume at the next time step.

# In[ ]:


#create dataframe series for t+1,t+2,t+3, to be used as y values, during Supervised Learning
#lookback = 5, means 5 values of TimeSeries (x) are used to predict the value at time t+1,t+2,t+3 (y)
def createSupervisedTrainingSet(dataset,lookback):

    df = DataFrame()
    x = dataset
    
    len_series = x.shape[0]

    df['t'] = [x[i] for i in range(x.shape[0])]
    #create x values at time t
    x=df['t'].values
    
    cols=list()
  
    df['t+1'] = df['t'].shift(-lookback)
    cols.append(df['t+1'])
    df['t+2'] = df['t'].shift(-(lookback+1))
    cols.append(df['t+2'])
    df['t+3'] = df['t'].shift(-(lookback+2))
    cols.append(df['t+3'])
    agg = concat(cols,axis=1)
    y=agg.values

    x = x.reshape(x.shape[0],1)

    len_X = len_series-lookback-2
    X=np.zeros((len_X,lookback,1))
    Y=np.zeros((len_X,3))
 
    for i in range(len_X):
        X[i] = x[i:i+lookback]
        Y[i] = y[i]

    return X,Y

look_back = 3
trainX, trainY = createSupervisedTrainingSet(train, look_back)
testX,testY = createSupervisedTrainingSet(test, look_back)


# In[ ]:


testY=testY.reshape(testY.shape[0],testY.shape[1])
trainY=trainY.reshape(trainY.shape[0],trainY.shape[1])
print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)


# In[ ]:


#Check the sample train X and train Y, and match with original time series data
print1 = trainY[13,:].reshape(1,-1)
print("Train X at index 13")
print(np.around((trainX[13,:,:])))
print("Train Y at index 13")
print(np.around((print1)))
print("Actual Data")
print(np.around((dataset[13:19])))        
#We used a lookback value of 5
#We inspect the X,Y values at a random index: 13
#As can be seen the 5 values of Time Series (Call Volume) from index 13 are being used as X to 
#predict the 3 values coming next (t+1,t+2,t+3)


# Use a Deep Learning technique with one hidden layer of 20 LSTM cells, outputting into 3 values, ie the predictions at time t+1,t+2,t+3. Input layer being 10 by 1 in size, for the 10 prior values of time series.

# In[ ]:


model = Sequential()
model.add(LSTM(16,activation='relu',return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(8, activation='relu'))
model.add(Dense(3))
myOptimizer = optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=0.01, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=myOptimizer)
history = model.fit(trainX, trainY, epochs=200,  validation_data=(testX,testY), batch_size=5, verbose=2)


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'], color=  'red')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


#Once the model is trained, use it to make a prediction on the test data
testPredict = model.predict(testX)
predictUnscaled = np.around((testPredict))
testYUnscaled = np.around((testY))
#print the actual and predicted values at t+3
print("Actual values of COVID 19 cases")
print(testYUnscaled[:,0])
print("Predicted values of COVID 19 cases")
print(predictUnscaled[:,0])


# Plot the predicted and actual values at time t+1,t+2,t+3

# In[ ]:


pyplot.plot(testPredict[:,0], color='red')
pyplot.plot(testY[:,0])
pyplot.legend(['Predicted','Actual'])
pyplot.title('Actual vs Predicted at time t+1')
pyplot.show()


# In[ ]:


#Evaluate the RMSE values at t+1,t+2,t+3 to compare with other approaches, and select the best approach
def evaluate_forecasts(actuals, forecasts, n_seq):
    	for i in range(n_seq):
            actual = actuals[:,i]
            predicted = forecasts[:,i]
            rmse = sqrt(mean_squared_error(actual, predicted))
            print('t+%d RMSE: %f' % ((i+1), rmse))
        
evaluate_forecasts(testYUnscaled, predictUnscaled,3)


# **Grid Search of parameters**
# We can improve the DeepLearning approach further by using Grid Search of Neural Network parameters using sklearn wrapper for Keras, KerasRegressor, and GridSearchCV. 

# In[ ]:


dataset_cases[:,0].astype(int)


# In[ ]:


from statsmodels.tsa.stattools import adfuller
from numpy import log
result = adfuller(dataframe_wton['cases'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])


# In[ ]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

# Original Series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(dataframe_wton['cases']); axes[0, 0].set_title('Original Series')
plot_acf(dataframe_wton['cases'], ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(dataframe_wton['cases'].diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(dataframe_wton['cases'].diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(dataframe_wton['cases'].diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(dataframe_wton['cases'].diff().diff().dropna(), ax=axes[2, 1])


# In[ ]:


# PACF plot of 1st differenced series
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(dataframe_wton['cases'].diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,5))
plot_pacf(dataframe_wton['cases'].diff().dropna(), ax=axes[1])

plt.show()


# In[ ]:


plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(dataframe_wton['cases'].diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,1.2))
plot_acf(dataframe_wton['cases'].diff().dropna(), ax=axes[1])

plt.show()


# In[ ]:


dataframe_wton = dataframe[(dataframe.county=='Washington') & (dataframe.state=='Oregon')]
dataframe_wton.index=dataframe_wton['pd_date']
dataframe_wton = dataframe_wton.iloc[:,4:6]


# In[ ]:


from statsmodels.tsa.arima_model import ARIMA

# 1,1,2 ARIMA Model
model = ARIMA(dataframe_wton['cases'].astype(float), order=(1,1,2))
model_fit = model.fit(disp=0)
print(model_fit.summary())


# In[ ]:


# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()


# In[ ]:


# Actual vs Fitted
model_fit.plot_predict(dynamic=False)
plt.show()

