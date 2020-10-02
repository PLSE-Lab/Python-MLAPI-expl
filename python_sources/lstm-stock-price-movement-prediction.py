#!/usr/bin/env python
# coding: utf-8

# ### To predict the movement of stock prices, LSTM model is used. Here, we are only predicting the movement of closing price but the similar approach can be followed for other columns as well.
# ### The model is trained on the stock prices of AAL company which is then tested on other companies.
# ### As this is a time series prediction, we are considering the prices of past 60 days to predict the future price. 

# In[152]:


# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[153]:


# loading the stock prices of all companies in a dataframe
dataset = pd.read_csv('../input/all_stocks_5yr.csv')


# In[154]:


dataset.head()


# In[155]:


dataset.info()


# In[156]:


# getting the list of all companies 
companies = dataset.Name.unique()
companies


# In[157]:


# since AAL company is used for training, we are creating a new dataframe with AAL parameters
stock = dataset.loc[dataset['Name'] == 'AAL']
stock.info()


# In[158]:


stock.head()


# In[159]:


# creating an array with closing prices
training_set = stock.iloc[:, 4:5].values


# In[160]:


# normalizing the values
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set_scaled = sc.fit_transform(training_set)


# In[161]:


training_set_scaled.shape


# ### In the below cell, we are creating 2 arrays, x_train and y_train.
# * x_train stores the values of closing prices of past 60(or as specified in timestamp) days
# * y_train stores the values of closing prices of the present day

# In[162]:


x_train = []
y_train = []
timestamp = 60
length = len(training_set)
for i in range(timestamp, length):
    x_train.append(training_set_scaled[i-timestamp:i, 0])
    y_train.append(training_set_scaled[i, 0])
    
x_train = np.array(x_train)
y_train = np.array(y_train)


# In[163]:


print (x_train[0])
print ('\n')
print (y_train[0])


# In[164]:


x_train.shape


# In[165]:


x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape


# In[166]:


from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

model = Sequential()

model.add(LSTM(units = 92, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 92, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 92, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 92, return_sequences = False))
model.add(Dropout(0.2))

model.add(Dense(units = 1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[167]:


model.fit(x_train, y_train, epochs = 25, batch_size = 32)


# ### Now the model is trained. We will test the performance of our model by plotting the predicted stock prices and actual stock prices of other companies.

# In[168]:


test_set = dataset.loc[dataset['Name'] == 'SIG']   # change CBS to whatever company from the list
test_set = test_set.loc[:, test_set.columns == 'close']


# In[169]:


# storing the actual stock prices in y_test starting from 60th day as the previous 60 days are used to predict the present day value.
y_test = test_set.iloc[timestamp:, 0:].values


# In[170]:


# storing all values in a variable for generating an input array for our model 
closing_price = test_set.iloc[:, 0:].values
closing_price_scaled = sc.transform(closing_price)


# In[171]:


# the model will predict the values on x_test
x_test = [] 
length = len(test_set)

for i in range(timestamp, length):
    x_test.append(closing_price_scaled[i-timestamp:i, 0])
    
x_test = np.array(x_test)
x_test.shape


# In[172]:


x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
x_test.shape


# In[173]:


# predicting the stock price values
y_pred = model.predict(x_test)
predicted_price = sc.inverse_transform(y_pred)


# In[174]:


# plotting the results
plt.plot(y_test, color = 'blue', label = 'Actual Stock Price')
plt.plot(predicted_price, color = 'red', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# In[ ]:




