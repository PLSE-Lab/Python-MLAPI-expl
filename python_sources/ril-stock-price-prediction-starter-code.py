#!/usr/bin/env python
# coding: utf-8

# # RIL Stock Price Prediction : Starter Code
# 
# 

# # Import the libraries

# In[ ]:



import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from datetime import datetime
import math


from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# # Read the data

# In[ ]:




ril_price= pd.read_csv("../input/reliance-industries-ril-share-price-19962020/Reliance Industries 1996 to 2020.csv")
#Show the data 
ril_price


# Lots of rows have NaN value. lets delete those

# In[ ]:


ril_price=ril_price.dropna()
ril_price


# Even after removing NaN values, we have 2200+ rows and 9+ years of data. good enough for the analysis

# In[ ]:


ril_price.info()


# date is a string object not a date. lets fix this

# In[ ]:


ril_price["Date"]=pd.to_datetime(ril_price["Date"], format="%d-%m-%Y")


ril_price["Date"]

ril_price.set_index('Date', inplace=True)
ril_price.info()


# In[ ]:


ril_price.describe()


# # Create a Chart to visualize the data.

# In[ ]:


#Visualize the closing price history
plt.figure(figsize=(16,8))
plt.title('Reliance Industries Close Price History')
plt.plot(ril_price['Close'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price INR',fontsize=18)
plt.show()


# In[ ]:


#Create a new dataframe with only the 'Close' column
data = ril_price.filter(['Close'])
#Converting the dataframe to a numpy array
dataset = data.values
#Get /Compute the number of rows to train the model on
training_data_len = math.ceil( len(dataset) *.8) 


# # Try this Out!
# 
# Create a time series model to predict the stock price!

# Want some inspiration?
# 
# Look at:
# 
# 1) [RIL Stock Price Prediction- LSTM](https://www.kaggle.com/kmldas/ril-stock-price-prediction-lstm)
# 
# 2) [Reliance: Technical Analysis: Bollinger Bands, MA](https://www.kaggle.com/kmldas/reliance-technical-analysis-bollinger-bands-ma)
