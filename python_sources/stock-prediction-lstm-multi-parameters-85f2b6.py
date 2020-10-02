#!/usr/bin/env python
# coding: utf-8

# *by Alex Koo (2019)*
# *Alex Koo is a director at Phillip Private Equity*
# 
# This is a simple example to demonstrate the concept of using LSMT (a type of AI model) to predict share prices. The example is relatively simple. LSTM is suitable for time series data like in stock trading. For example, when we look at today's share prices, we not only look at the current PE, but also how the PE of the stock has changed over time. In another word, we need to remember the behavious of its PE over a period of time (memory), not just a static view of today's PE.
# 
# For a start, we load in all the libraries that are required. As you can see, these are Open-Source libraries, which means you do not have to be an expert scientist to use AI. The complex mathematics are solved by smart and very intelligence people working round the clock. Today, AI application programmers do not have to do the heavy lifting. The libraries, whether they are paid, free or trained, can be used like tools to solve real life problems. As long as your know how to use the tools. Much of the current advancement in AI is Deep Learning. The mathematics involved are complexed, but you only need to know how to use the tools in order to solve real life problems. Analogy is like electricity. Thomas Edison invented electricity. The major developments happen when engineers start to harness the power of electricity and build businesses around it to solve real life solutions.
# 

# In[ ]:



import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from tensorflow.keras.optimizers import SGD
import math
from sklearn.metrics import mean_squared_error

# Some functions to help out with
def plot_predictions(test,predicted,cur):
    plt.plot(test, color='red',label='Actual Stock Price')
    plt.plot(predicted, color='blue',label='Predicted Stock Price')
    plt.plot(cur, color='orange',label='Current Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

def return_rmse(test,predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {}.".format(rmse))
    


# We need the stock data for the AI engeine. I have downloaded some data from the Bloomberg terminal. There are manh other sources of stock data which you can download like Yahoo Finance. In this .csv (comma separate, can open with Excel) file, we pick a few parameters like Prices, and Analyst Target Price.
# If you sucessfully loaded the data file, the following codes will display a summary of the data file.

# Next, we create a new field which we try to predict by using the dataframe.shift() function. In this case, we are trying to predict the price in 10 days time. Few free to change the parameters later on.
# 
# We create a new last column 'PREDICT' which is the parameter we want to predict. In this case, it is the price in 10 days time, using the .shift method.

# In[ ]:


import pandas as pd
dataset = pd.read_csv("../input/msft.csv",index_col='Dates', parse_dates=['Dates'], dayfirst =True)

dataset['PREDICT'] = dataset['PX_LAST'].shift(-10) #trying to predict 10 days ahead, can change this to any days.
df=dataset.dropna()
df=df.iloc[:][:]
print(df.head())
print(df.describe())
size_data = len(df)
#split data_set
train_ratio=10
size_train = int(size_data*(1-train_ratio/100))
size_test = int(size_data*(train_ratio/100))

print('size of dataset:' , size_data)
print('size_train:', size_train)
print('size_train:', size_test)


# Next we have to split to data into 2 sets. One set for training and the rests for testing. The percentage is defined by . 
# 10 means that 10% of data set will be used for testing. After excuting the code below, you will see the graph of the data that are used for training and testing in different colors.

# In[ ]:



# Checking for missing values
# select column
#Dates	PX_LAST	PX_VOLUME	VOLATILITY_90D	BEST_ANALYST_RATING

selected_column_list = ['PX_LAST'
                        #,'VOLATILITY_90D'
                        ,'BEST_ANALYST_RATING'
                        ,'PREDICT']

num_fields = len(selected_column_list)
num_training_fields = num_fields -1

df = df[selected_column_list]

df_training_set = df[selected_column_list][:size_train]
df_test_set = df[selected_column_list][size_train:]

training_set = df_training_set.values
test_set     = df_test_set.values

size_train = len(df_training_set)
size_test  = len(df_test_set)


# Next, we need to pre-process the data by normaling the parameters. 
# 

# In[ ]:


from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

plt.clf()
plt.plot(df_training_set['PX_LAST'],color='red') 
plt.plot(df_test_set["PX_LAST"],color='blue')

plt.legend(['Training set','Test set'])
plt.title('MSFT stock price')
plt.show()

# Scaling the training set
sc = MinMaxScaler(feature_range=(0,1))
#fit the entire dataset
sc.fit(df) 

training_set_scaled = sc.transform(training_set)
test_set_scaled = sc.transform(test_set)


# In[ ]:


X_train = []
y_train = []
for i in range(60,len(training_set)):
    X_train.append(training_set_scaled[i-60:i,:-1])
    y_train.append(training_set_scaled[i,-1]) # last column is actual price to predict
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping X_train for efficient modelling
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],num_training_fields))


# The LSTM architecture
regressor = Sequential()
# First LSTM layer with Dropout regularisation
regressor.add(LSTM(units=20, return_sequences=True, input_shape=(X_train.shape[1],num_training_fields)))
regressor.add(Dropout(0.2))
# Second LSTM layer
regressor.add(LSTM(units=40, return_sequences=True))
regressor.add(Dropout(0.2))
# Third LSTM layer
regressor.add(LSTM(units=40, return_sequences=True))
regressor.add(Dropout(0.2))
# Fourth LSTM layer
regressor.add(LSTM(units=20))
regressor.add(Dropout(0.2))
# The output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='rmsprop',loss='mean_squared_error')
# Fitting to the training set
regressor.fit(X_train,y_train,epochs=10,batch_size=64)

# Now to get the test set ready in a similar way as the training set.
# The following has been done so forst 60 entires of test set have 60 previous values which is impossible to get unless we take the whole 
# 'High' attribute data for processing
#df = pd.concat((df_training_set, df_test_set),axis=0)


inputs = df.iloc[:][len(training_set)-60:].values

inputs = inputs.reshape(-1,num_fields)
inputs  = sc.transform(inputs)


# We then plot out the test data, and predicted data. 
# At any time, you have the current price, predicted price, and the actual price in 10 days. As you can see, the system is not accurate. Try using more parameters which you think has a influence of the predict share price.
# 
# Any item which has not been added is to monitor the errors, and simulate a particular tradiing strategy and see if we can make money from it. E.g. to buy a particular amount of shares if the predicted price is more than 10% of current price.
# 
# Some models try to precict the 1 day price, but that is not very meaningful to be used in an investment strategy.
# 

# In[ ]:


# Preparing X_test and predicting the prices
X_test = []
y_test = []
for i in range(60,len(inputs)):
    X_test.append(inputs[i-60:i,:-1])
    y_test.append(inputs[i,-1])
X_test = np.array(X_test)
y_test = np.array(y_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],num_training_fields))

predicted_stock_price_sc = regressor.predict(X_test)

#compute inverse scale
# delete last column
predict_test_set_scaled = test_set_scaled[:,:-1]
#add last column with predicted value
predict_test_set_scaled = np.concatenate((predict_test_set_scaled,predicted_stock_price_sc), axis=1 )
predict_test_set = sc.inverse_transform(predict_test_set_scaled)



# Visualizing the results for LSTM
#test
#actual
#curent

plot_predictions(test_set[:,-1],predict_test_set[:,-1],test_set[:,0])

# Evaluating our model
return_rmse(test_set[:,-1],predict_test_set[:,-1])

