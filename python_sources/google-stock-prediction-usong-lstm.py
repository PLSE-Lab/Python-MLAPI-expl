#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_csv('../input/Google.csv')


# Choosing 'Close' as our target variable

# 

# In[3]:


data=data.iloc[:,4].values
data=pd.DataFrame(data)


# Splitting dataset into train and test

# In[4]:


train_data=data.iloc[0:2500].values
test_data=data.iloc[2500: ,].values
test_data=pd.DataFrame(test_data)
train_data=pd.DataFrame(train_data)


# Scaling the train dataset

# In[5]:


from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range =(0,1))
train_data_scaled=sc.fit_transform(train_data)
train_data_scaled=pd.DataFrame(train_data_scaled)


# In[6]:


x_train=[]
y_train=[]


# Selecting 90 values for data structure

# In[7]:


for i in range(90,2500):
    x_train.append(train_data_scaled.iloc[i-90:i,0])
    y_train.append(train_data_scaled.iloc[i,0])


# Reshaping the data structure

# In[8]:


x_train,y_train=np.array(x_train), np.array(y_train) 
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1 ))   
    


# In[9]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# Initializing RNN

# In[10]:


regressor=Sequential()


# 1st Layer

# In[11]:


regressor.add(LSTM(units = 50,return_sequences = True,input_shape=(x_train.shape[1],1)))
regressor.add(Dropout(0.2))   


# 2nd Layer

# In[12]:


regressor.add(LSTM(units = 50,return_sequences = True,))
regressor.add(Dropout(0.2))      


# 3rd Layer

# In[13]:


regressor.add(LSTM(units = 50,return_sequences = True,))
regressor.add(Dropout(0.2))     


# 4th Layer

# In[14]:


regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))    


# Output Layer

# In[15]:


regressor.add(Dense(units=1))


# In[16]:


regressor.compile(optimizer='adam',loss='mean_squared_error')


# In[17]:


regressor.fit(x_train,y_train,epochs=50,batch_size=32)


# In[18]:


data.head()


# In[19]:


print(data)


# In[55]:


input=data.iloc[len(data)-len(test_data)-90: ].values


# In[56]:


print(input)


# In[29]:


type(input)


# In[57]:


input=sc.transform(input)


# Creating data structure for test set

# 

# In[58]:


input=pd.DataFrame(input)


# In[59]:




x_test=[]

for i in range(90,len(input)):
    x_test.append(input.iloc[i-90:i, 0])
        


# In[60]:


x_test=np.array(x_test)
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))


# In[61]:


predicted_stock_price=regressor.predict(x_test)
predicted_stock_price=sc.inverse_transform(predicted_stock_price)


# In[62]:


print(predicted_stock_price)


# In[63]:


pred=pd.DataFrame(predicted_stock_price)


# In[64]:


plt.plot(test_data,color='red',label='real stock price')
plt.plot(predicted_stock_price,color='blue',label='predicted stock price')
plt.title('goole stock price')
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
plt.show()


# In[65]:


import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(test_data, predicted_stock_price))


# In[66]:


print(rmse)

