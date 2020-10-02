#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import IPython
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


fundamentals= pd.read_csv('../input/nyse/fundamentals.csv')


# In[ ]:


fundamentals.head()


# In[ ]:


pricesa_data= pd.read_csv('../input/nyse/prices-split-adjusted.csv')
pricesa_data.head()


# In[ ]:


price_data= pd.read_csv('../input/nyse/prices.csv')
price_data.head()


# In[ ]:


security_data= pd.read_csv('../input/nyse/securities.csv')
security_data.head()


# In[ ]:


#Industry-wise tabulation of number of companies given in the fundamentals data
plt.figure(figsize=(15, 6))
ax = sns.countplot(y='GICS Sector', data=security_data)
plt.xticks(rotation=45)


# Since we just have to predict the stock price for next day, we'll need only 'price_data' because that has the information regarding stock price and that's our point of concern.

# In[ ]:


price_data.info()


# In[ ]:


len(price_data.symbol.unique())


# Now, we'll focus on one company for which we want to predict the stock price for. Enter the ticker code of that company and we'll use that for all our analysis and prediction.

# In[ ]:


#let's make a function that'll plot the opening and closing chart for the company chosen

def chart(ticker):
    global closing_stock
    global opening_stock
    f, axs = plt.subplots(2,2,figsize=(16,8))
    plt.subplot(212)
    company = price_data[price_data['symbol']==ticker]
    company = company.open.values.astype('float32')
    company = company.reshape(-1, 1)
    opening_stock = company
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel(ticker + " open stock prices")
    plt.title('prices Vs Time')
    
    plt.plot(company , 'g')
    
    plt.subplot(211)
    company_close = price_data[price_data['symbol']==ticker]
    company_close = company_close.close.values.astype('float32')
    company_close = company_close.reshape(-1, 1)
    closing_stock = company_close
    plt.xlabel('Time')
    plt.ylabel(ticker + " close stock prices")
    plt.title('prices Vs Time')
    plt.grid(True)
    plt.plot(company_close , 'b')
    
    plt.show()
ticker = input("Enter the ticker of the company you want to see the graph for -")
chart(ticker)


# In[ ]:


#make a data-frame that'll have details for the company chosen
ticker_data= price_data[price_data['symbol']==ticker]


# In[ ]:


#this checks the amount of data we've for the company chosen
train_dates=list(ticker_data.date.unique())
print(f"Period : {len(ticker_data.date.unique())} days")
print(f"From : {ticker_data.date.min()} To : {ticker_data.date.max()}")


# In[ ]:


#importing libraries
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense , BatchNormalization , Dropout , Activation
from keras.layers import LSTM , GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam , SGD , RMSprop


# In[ ]:


data=ticker_data.copy()


# In[ ]:


data.head()


# The profit or loss calculation is usually determined by the closing price of a stock for the day, hence we will consider the closing price as the target variable and drop the other columns.

# In[ ]:


data.drop(['symbol','open','low','high','volume'],axis=1,inplace=True)


# In[ ]:


data.head()


# Now, the dtype of date is object. We'll convert that into date-time and will make it as the index for our dataframe.

# In[ ]:


data['date'] = pd.to_datetime(data['date'], infer_datetime_format=True)


# In[ ]:


data.index=data.date


# In[ ]:


data.drop('date', axis=1,inplace=True)


# In[ ]:


data.head()


# In[ ]:


dataset=data.values


# In[ ]:


#scale the dataset
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)


# In[ ]:


#we'll use 80% of the data as training data 
train = int(len(dataset) * 0.80)
test = len(dataset) - train


# In[ ]:


print(train, test)


# In[ ]:


train= dataset[:train]
test = dataset[len(train):]


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


#I'll use past two days data to predict the price for next day. I tried a few numbers and this was giving least error. Therefore, I thought of using this.
x_train, y_train = [], []
for i in range(len(train)-2):
    x_train.append(dataset[i:i+2,0])
    y_train.append(dataset[i+2,0])
x_train, y_train = np.array(x_train), np.array(y_train)


# In[ ]:


x_train.shape


# In[ ]:


x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
x_train.shape


# In[ ]:


x_train.shape


# In[ ]:


x_test = []
y_test=[]
for i in range(len(test)-2):
    x_test.append(dataset[len(train)-2+i:len(train)+i,0])
    y_test.append(dataset[len(train)+i,0])
x_test = np.array(x_test)
y_test = np.array(y_test)


# In[ ]:


x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
x_test.shape


# I will use LSTM to predict the stock price because LSTM stores the past information in predicting the future.

# In[ ]:


#I've used a LSTM model to predict the stock price. i checked for various other models but this was giving the least error. Therefore, I've used that
model= Sequential([
                   LSTM(256, input_shape=(x_train.shape[1],1), return_sequences=True),
                   Dropout(0.4),
                   LSTM(256),
                   Dropout(0.2),
                   Dense(16, activation='relu'),
                   Dense(1)
])
print(model.summary())


# In[ ]:


model.compile(loss='mean_squared_error', optimizer=Adam(lr = 0.0005) , metrics = ['mean_squared_error'])


# In[ ]:


history = model.fit(x_train, y_train, epochs=40 , batch_size = 128, validation_data=(x_test, y_test))


# In[ ]:


#summarize history for error
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('model mean squared error')
plt.ylabel('Error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()


# In[ ]:


#using the model for x_test and then converting the data to normal price using inverse transform
predicted_price= model.predict(x_test)
predicted_price = scaler.inverse_transform(predicted_price)


# In[ ]:


len(predicted_price)


# In[ ]:


predicted_price =np.array(predicted_price)
predicted_price.shape


# In[ ]:


y_test.shape


# In[ ]:


#checking the score for our data
def model_score(model, X_train, y_train, X_test, y_test):
    trainScore = model.evaluate(X_train, y_train, verbose=0)
    print('Train Score: %.5f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))
    testScore = model.evaluate(X_test, y_test, verbose=0)
    print('Test Score: %.5f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
    return trainScore[0], testScore[0]

model_score(model, x_train, y_train , x_test, y_test)


# Since the MSE and RMSE is quite less for both: the training as well as the test data, our model does a great job.

# In[ ]:


predicted_price[:10]


# In[ ]:


y_test = y_test.reshape(y_test.shape[0] , 1)
y_test = scaler.inverse_transform(y_test)
y_test[:10]


# In[ ]:


#comparing the first 10 values of prediction for our data
diff = predicted_price-y_test
diff[:10]


# In[ ]:


#plotting the courves for the actual test values and the predicted values. 
#The actual values are represented by the blue line and the predicted value by the red line
print("Red - Predicted Stock Prices  ,  Blue - Actual Stock Prices")
plt.rcParams["figure.figsize"] = (15,7)
plt.plot(y_test , 'b')
plt.plot(predicted_price , 'r')
plt.xlabel('Time')
plt.ylabel('Stock Prices')
plt.title('Check the accuracy of the model with time')
plt.grid(True)
plt.show()


# The final plot gives the inference that our model has done pretty well in predicting the stock price for the chosen company(i.e. Microsoft).
