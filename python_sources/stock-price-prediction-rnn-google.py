#!/usr/bin/env python
# coding: utf-8

# Here we will be working with Google stock Price data.We will be using five year of google Stock price data and try to predict the future stock price using RNN.This Kernel is a work in process.If you like my work please do vote.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **Importing Python Modules**

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# **Importing the dataset** 

# In[ ]:


dataset_train=pd.read_csv('../input/googledta/trainset.csv')
dataset_train.tail()


# In[ ]:


dataset_test=pd.read_csv('../input/google-test/Google_Stock_Price_Test.csv')
dataset_test.head()


# **Part1:Data Preprocessing **

# In[ ]:


training_set=dataset_train.iloc[:,1:2].values


# We are considering the open stock price for prediction.Taking the range 1:2 means only column of Open Price is selected and we also have an numpy array

# **Applying Feature Scaling **

# In[ ]:


import matplotlib.pyplot as plt
from PIL import Image
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
img=np.array(Image.open('../input/feature-scaling/Normalization.PNG'))
fig=plt.figure(figsize=(10,10))
plt.imshow(img,interpolation='bilinear')
plt.axis('off')
plt.show()


# We can used Feature scaling Method like Standardisation and Normalization on our Dataset.For RNN we prefer Normalization as it work well if Sigmoid function is used for activation in the output layer

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
training_set_scaled=sc.fit_transform(training_set)


# In[ ]:


training_set_scaled.shape


# Now we have the stock prices in ranged of 0 to 1 due to scaling.

# **Creating a data Structure with 60 timesteps and one output **
# 
# We will be taking the reference of past 60 days of data to predict the future stock price.It is observed that taking 60 days of past data gives us best results.In this data set 60 days of data means 3 months of data.Every month as 20 days of Stock price. X train will have data of 60 days prior to our date and y train will have data of one day after our date

# In[ ]:


X_train=[]
y_train=[]
for i in range(60,1259):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0]) 
X_train,y_train=np.array(X_train),np.array(y_train)


# **Reshaping the data**

# In[ ]:


X_train.shape[0]


# In[ ]:


X_train.shape[1]


# In[ ]:


X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))


# In[ ]:


X_train.shape


# Now we have the right shape for our data in three dimensons.
# 
# 1-Number of stock prices -1199
# 
# 2-Number of time steps -60
# 
# 3.Number of Indicator -1

# **Part2:Building the RNN** 

# **Importing the Keras packages**

# In[ ]:


from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import LSTM
from keras.layers import Dropout


# **Initalize RNN **

# In[ ]:


regressor=Sequential()


# * **Adding the first LSTM Layer and Dropout regularization**

# In[ ]:


regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))


# **Adding second layer of LSTM and dropout regularization**

# In[ ]:


regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))


# > **Adding third layer of LSTM and dropout regularization**

# In[ ]:


regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))


# **Adding third layer of LSTM and dropout regularization**

# In[ ]:


regressor.add(LSTM(units=50,return_sequences=False))
regressor.add(Dropout(0.2))


# **Adding the output Layer **

# In[ ]:


regressor.add(Dense(units=1))


# **Compiling the RNN**

# In[ ]:


regressor.compile(optimizer='adam',loss='mean_squared_error')


# > **Fitting the RNN to training set**

# In[ ]:


regressor.fit(X_train,y_train,epochs=100,batch_size=32)


# **Part3:Making Predictions and Making Visulization**

# Getting the real stock price of 2017

# In[ ]:


dataset_test=pd.read_csv('../input/google-test/Google_Stock_Price_Test.csv')
dataset_test.head()


# In[ ]:


real_stock_price=dataset_test.iloc[:,1:2].values


# In[ ]:


dataset_total=pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
inputs=dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs=inputs.reshape(-1,1)
inputs=sc.transform(inputs)


# In[ ]:


X_test=[]
for i in range(60,80):
    X_test.append(inputs[i-60:i,0])
X_test =np.array(X_test)


# In[ ]:


X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_stock_price=regressor.predict(X_test)
predicted_stock_price=sc.inverse_transform(predicted_stock_price)


# In[ ]:


predicted_stock_price.shape


# In[ ]:


real_stock_price.shape


# **Visualizing the results**

# In[ ]:


plt.plot(real_stock_price,color='red',label='Real Google Stock Price')
plt.plot(predicted_stock_price,color='blue',label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


# In[ ]:




