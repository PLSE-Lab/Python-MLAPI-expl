#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


#!pip install iexfinance


# In[3]:


# Machine learning
from sklearn.svm import SVC
from sklearn.metrics import scorer
from sklearn.metrics import accuracy_score

# For data manipulation
import pandas as pd
import numpy as np

# To plot
import matplotlib.pyplot as plt
import seaborn


# In[4]:


# # Fetch the Data
# from iexfinance import get_historical_data 
# from datetime import datetime

# start = datetime(2017, 1, 1) # starting date: year-month-date
# end = datetime(2018, 1, 1) # ending date: year-month-date

# Df = get_historical_data('SPY', start=start, end=end, output_format='pandas')          
# Df= Df.dropna()
# Df = Df.rename (columns={'open':'Open', 'high':'High','low':'Low', 'close':'Close'})

# Df.Close.plot(figsize=(10,5))
# plt.ylabel("S&P500 Price")
# plt.show()


# In[5]:


Df = pd.read_csv('../input/eur_dol_indicators5mnV2.csv')[['mid.o','mid.h','mid.l','mid.c']]


# In[6]:


Df.head()


# In[7]:


Df = Df.tail(10000)


# In[8]:


Df= Df.dropna()
Df = Df.rename (columns={'mid.o':'Open', 'mid.h':'High','mid.l':'Low', 'mid.c':'Close'})

Df.Close.plot(figsize=(10,5))
plt.ylabel("eur/dol price")
plt.show()


# # Determine the correct trading signal
# 
# If tomorrow's price is greater than today's price then we will buy the S&P500 index, else we will sell the S&P500 index. We will store +1 for buy signal and -1 for sell signal in Signal column. y is a target dataset storing the correct trading signal which the machine learning algorithm will try to predict.

# In[9]:


y = np.where(Df['Close'].shift(-1) > Df['Close'],1,-1)


# # Creation of predictors datasets
# 
# The X is a dataset that holds the variables which are used to predict y, that is, whether the S&P500 index price will go up (1) or go down (-1) tomorrow. The X consists of variables such as 'Open - Close' and 'High - Low'. These can be understood as indicators based on which the algorithm will predict tomorrow's trend. Feel free to add mroe indicators and see the performance.

# In[10]:


Df['Open-Close'] = Df.Open - Df.Close
Df['High-Low'] = Df.High - Df.Low
X=Df[['Open-Close','High-Low']]
X.head()


# # Test and train data set split
# 
# Now, we will split data into training and test data set. 
# 
# 1. First 80% of data is used for training and remaining data for testing.
# 2. X_train and y_train are training dataset.
# 3. X_test and y_test are test dataset.

# In[11]:


split_percentage = 0.8
split = int(split_percentage*len(Df))

# Train data set
X_train = X[:split]
y_train = y[:split] 

# Test data set
X_test = X[split:]
y_test = y[split:]


# # Support Vector Classifier (SVC)
# 
# We will use SVC() function from sklearn.svm.SVC library for the classification and create our classifier model using fit() method on the training data set.

# In[12]:


cls = SVC().fit(X_train, y_train)


# # Classifier accuracy
# We will compute the accuarcy of the algorithm on the train and test data set, by comparing the actual values of Signal with the predicted values of Signal. The function accuracy_score() will be used to calculate the accuracy.
# 
# <B>Syntax:</B> accuracy_score(<font color=blue>target_actual_value</font>,<font color=blue>target_predicted_value</font>)
# 1. <font color=blue>target_actual_value:</font> correct signal values
# 2. <font color=blue>target_predicted_value:</font> predicted signal values

# In[13]:


accuracy_train = accuracy_score(y_train, cls.predict(X_train))
accuracy_test = accuracy_score(y_test, cls.predict(X_test))

print('\nTrain Accuracy:{: .2f}%'.format(accuracy_train*100))
print('Test Accuracy:{: .2f}%'.format(accuracy_test*100))


# An accuracy of 50%+ in test data suggests that the classifier model is effective.

# # Prediction
# 
# ### Predict signal 
# 
# We will predict the signal (buy or sell) for the test data set, using the cls.predict() fucntion.
# 
# ### Compute returns in test period
# 
# We will compute the strategy returns based on the predicted signal, and then save it in the column 'Strategy_Return' and plot the cumulative strategy returns.

# # Strategy Implementation
# 

# In[14]:


Df['Predicted_Signal'] = cls.predict(X)
# Calculate log returns
Df['Return'] = np.log(Df.Close.shift(-1) / Df.Close)*100
Df['Strategy_Return'] = Df.Return * Df.Predicted_Signal
Df.Strategy_Return.iloc[split:].cumsum().plot(figsize=(10,5))
plt.ylabel("Strategy Returns (%)")
plt.show()


# In[ ]:





# In[ ]:




