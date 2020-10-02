#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install tensorflow-gpu')


# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


# In[ ]:


## import the data from the quandl
get_ipython().system('pip install quandl')
import quandl


# In[ ]:


quandl.ApiConfig.api_key = 'zuiQMfguw3rRgLvkCzxk'


# In[ ]:


df = quandl.get('WIKI/GOOGL')


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


##checking is there null value
df.isnull().sum()


# In[ ]:


df.corr()[['Adj. Close']]


# In[ ]:


## dropping the split ratio adj.Vlume Volume EX-Divident
df=df.drop('Volume',axis=1)


# In[ ]:


df=df.drop('Adj. Volume',axis=1)
df=df.drop('Split Ratio',axis=1)
df=df.drop('Ex-Dividend',axis=1)


# In[ ]:


df.head()


# In[ ]:


df.corr()['Adj. Close'].plot()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


X = df.drop('Adj. Close',axis=True)
y = df[['Adj. Close']]


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=.2)
result = y_test
scaler = MinMaxScaler()
scaler.fit(x_train)
xtrain_t = scaler.transform(x_train)
scaler.fit(x_test)
xtest_t = scaler.transform(x_test)
scaler.fit(y_train)
y_train =scaler.transform(y_train) # we transform the y so after predict we have to inverse transeform it
scaler.fit(y_test)
y_test =scaler.transform(y_test) # we transform the y so after predict we have to inverse transeform it


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Flatten


# In[ ]:





# In[ ]:


x_train = np.reshape(xtrain_t, (xtrain_t.shape[0],xtrain_t.shape[1],1))
x_test = np.reshape(xtest_t, (xtest_t.shape[0],xtest_t.shape[1],1))


# In[ ]:


print (x_train.shape)
print (x_test.shape)
print (y_train.shape)
print (y_test.shape)


# In[ ]:


regressor = Sequential()
regressor.add(LSTM(units = 50,return_sequences = True))


# In[ ]:


regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(0.2))


# In[ ]:


regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))


# In[ ]:


regressor.add(Dense(units = 1))


# In[ ]:


regressor.compile(optimizer = 'adam',loss = 'mean_squared_error',metrics=['accuracy'])


# In[ ]:


regressor.fit(x_train,y_train,epochs = 100)


# In[ ]:


y_pred = regressor.predict(x_test)


# In[ ]:


output = scaler.inverse_transform(y_pred)


# In[ ]:


real_output = []
for item in output:
  real_output.append((item[0]))


# In[ ]:


actual_output = []
for item in result['Adj. Close']:
  actual_output.append((item))


# In[ ]:


from sklearn.metrics import mean_squared_error


# In[ ]:


mse = mean_squared_error(actual_output,real_output)


# In[ ]:


mse


# In[ ]:


result['predited value'] = np.array(real_output)


# In[ ]:


result


# In[ ]:


import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,20)
result.plot()


# In[ ]:




