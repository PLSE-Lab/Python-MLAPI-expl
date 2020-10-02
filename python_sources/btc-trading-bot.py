#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


#import the libraries
import matplotlib.pyplot as plt
from keras.models import Sequential;
from keras.layers import Dense , LSTM , Dropout;
from sklearn.model_selection import train_test_split


# In[ ]:


raw = pd.read_csv("/kaggle/input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv")
raw = raw.Open.dropna().values


# In[ ]:


scaler = np.max(raw)


# In[ ]:


aux = []

for i in range (0 , len(raw) , 1440 ):
    aux.append(raw[i]/scaler)


# We prepare the data , organizing it in groups of 10 consecutive days for x and the value for the following day for y 

# In[ ]:


t_step = 10
x , y = [] , []
for i in range(len(aux)-t_step-1):
    x.append(aux[i:i+t_step])
    y.append(aux[i+t_step])

x , x_test , y , y_test = train_test_split(x , y , test_size = 0.2 , shuffle = False )
x = np.array(x)
x_test = np.array(x_test)
y = np.array(y)
y_test = np.array(y_test)


# In[ ]:


model = Sequential([
    LSTM( 32 , input_shape=(1, t_step)  , return_sequences = True ) ,
    LSTM( 32 ) ,
    Dropout(0.2) , 
    Dense(1)  
])
model.compile(loss='mse' , optimizer='adam' )


# In[ ]:


def prepare_for_fitting(arr):
    return np.reshape(arr , ( arr.shape[0] , 1 ,  arr.shape[1] )   )   # reshape in order to fit the model

x = prepare_for_fitting(x)
x_test = prepare_for_fitting(x_test)


# In[ ]:


history = model.fit( x , y , epochs = 200)


# In[ ]:


#Visualize loss
plt.figure(figsize=(7,7));
plt.title("Loss over time");
plt.xlabel("Epoch");
plt.ylabel("Loss");
plt.plot(history.history['loss']);


# In[ ]:


preds = model.predict(x_test) * scaler


# Visualize the predictions next to the actual values

# In[ ]:


plt.figure( figsize=( 12 , 12 ) );
plt.plot(preds , c='red' , label='Predicted' , linewidth=4);
plt.plot(  y_test*scaler , c='blue' , label='Actual values' , linewidth=4);
plt.xlabel("Day number");
plt.ylabel("Value in $");
plt.legend();


# In[ ]:


y_test*=scaler


# Now let's simulate a trading "bot" that takes as input the value of BTC for the last 10 days and predicts the price for the next one.
# It starts with $1000 and it can buy BTC with all the money , sell al the BTC or do nothing for the day. For this purpose , we will be using the 
# testing data and the predictions already generated. 

# In[ ]:


money_over_time = []
money = initial = 4000
money_in_btc = False
growth = 1
threshold = 0.95      # we set a threshold value wich represents the price of the last day divided by the price predicted for tomowwor
                      # in order to make a profit from a transaction it should be smaller than 1 , but not too small because there can't be
                      # such a big change in price over one day , meaning it will never buy anything
    
    
for i in range(len(preds)) :
    money*=growth
    
    money_over_time.append(money)
    
    if  y_test[i] / preds[i] < threshold:            # check whether the predicted price for tomorrow is smaller than the expected price for tomorrow 
        money_in_btc = True                 # if so , buy all the BTC you can
        growth = y_test[i+1]/ y_test[i]
    else :                       
        money_in_btc = False                # otherwise we convert the BTC to money
        growth = 1 


# Now let's plot the data and see how much money it made

# In[ ]:


plt.figure(figsize=(10,10));
plt.plot(money_over_time);
print(f"Over the course of {len(money_over_time)} days our 'trading bot' has ${int(money)} and it started with ${initial}")
print(f"A ${int(money-initial)} difference , or {     '%.2f'%(money/initial*100 - 100)}% profit")


# In[ ]:




