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


# Load data below

# In[ ]:


import pandas as pd
data = pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv")


# Print the data content (5 top rows). 

# In[ ]:


print(data.head(5))


# Data has 32 columns where there are two type of hotels 
# 1. Resort hotel 
# 2. City hotel
# arrival time of guests is given in terms of year, month, week number, day of month, in different columns. 
# Guests stay 
# 

# Also, printing the list of columns in the data.

# In[ ]:


print(list(data.columns))


# Scanning through the unique values of columns.
# 

# In[ ]:


print("hotel type ", data['hotel'].unique())
print("hotel value counts ",data['hotel'].value_counts())
print("Reservation status ",data['reservation_status'].unique())
print("Reservation value counts ",data['reservation_status'].value_counts())
print("meal type ", data['meal'].unique())


# In[ ]:


data.describe()


# In[ ]:


data.info()


# As we see here, there are columns which has null values such as 'country', 'company', 'agent', 'children'. So let's remove these columns from the data at the moment, except 'children'.

# In[ ]:


data = data.drop(['country', 'company', 'agent'], axis=1)
print(list(data.columns))


# **Problem Formulation -- **
# 
# For hotel, predicting if the guest is going to cancel the reservation or not will be helpful. So let's see how does the various field correlate with the 'is_canceled' column.

# In[ ]:


cancel_corr = data.corr()["is_canceled"]
cancel_corr.abs().sort_values(ascending=False) 


# Looking at the correlations, we can ignore the least correlated columns. But, let's keep all the columns at the moment. 
# But before predicting the cancelations, we need to transform the categorical columns. For that, we will use 'LabelEncoder'.

# Let's check the categorical columns, first.

# In[ ]:


print("hotel-- ", data['hotel'].unique())
print("\narrival_date_month-- ",data['arrival_date_month'].unique())
print("\nmeal-- ", data['meal'].unique())                         
print("\nmarket_segment-- ", data['market_segment'].unique())                   
print("\ndistribution_channel--  ", data['distribution_channel'].unique())
print("\nreserved_room_type--  ", data['reserved_room_type'].unique())
print("\nassigned_room_type--  ", data['assigned_room_type'].unique())
print("\ndeposit_type--  ", data['deposit_type'].unique())
print("\ncustomer_type--  ", data['customer_type'].unique())
print("\nreservation_status--  ", data['reservation_status'].unique())
print("\nreservation_status_date--  ", data['reservation_status_date'].unique())


# In[ ]:


# Import label encoder 
from sklearn import preprocessing 
# label_encoder object knows how to understand word labels. 
L_Encoder = preprocessing.LabelEncoder() 

# applyting label encoder on the hotel types 
data['hotel'] = L_Encoder.fit_transform(data['hotel'])
# on meal options
data['meal'] = L_Encoder.fit_transform(data['meal'])
# on market segement
data['market_segment'] = L_Encoder.fit_transform(data['market_segment'])
# on distribution_channel
data['distribution_channel'] = L_Encoder.fit_transform(data['distribution_channel'])
# on reserved_room_type
data['reserved_room_type'] = L_Encoder.fit_transform(data['reserved_room_type'])
# assigned_room_type
data['assigned_room_type'] = L_Encoder.fit_transform(data['assigned_room_type'])
# deposit_type
data['deposit_type'] = L_Encoder.fit_transform(data['deposit_type'])
# customer_type
data['customer_type'] = L_Encoder.fit_transform(data['customer_type'])
# reservation_status
data['reservation_status'] = L_Encoder.fit_transform(data['reservation_status'])


# In[ ]:


# manually encoding the arrival date
data['arrival_date_month']=data['arrival_date_month'].map({'January':1, 'February': 2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7,'August':8, 'September':9, 'October':10, 'November':11, 'December':12})
print(data['arrival_date_month'])


# In[ ]:


print(data.info())


# In[ ]:


# dropping reservation status
data = data.drop(['reservation_status_date'], axis = 1)
# dropping all the rows with NA values
data = data.dropna(axis = 0)


# Normalizing the data is another necessary step. But we are not applying here at the moment.
# 
# Let's prepare for the prediction of 'is_canceled'.

# In[ ]:


# Make X and y vectors for the modeling
X = data.drop(['previous_cancellations'], axis = 1)
y = data['previous_cancellations']
# import all the necessary packages 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression, LogisticRegression,Ridge, Lasso
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#
regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm
y_pred = regressor.predict(X_test) #predicting the cancellations
#


# test the realiability of predicted cancellations.

# In[ ]:


from sklearn import metrics
import numpy as np
# the error rate in predictions using Linear Regression
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print(metrics.mean_squared_error(y_test, y_pred))
print(metrics.mean_absolute_error(y_test, y_pred))


# Let's try to reduce the mean square error by using Logistic regression.

# In[ ]:


regressor = LogisticRegression(random_state=0,solver = 'lbfgs')  
regressor.fit(X_train, y_train) #training the algorithm
y_pred = regressor.predict(X_test) #predicting the cancellations


# In[ ]:


# the error rate in predictions using Logistics Regression
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print(metrics.mean_squared_error(y_test, y_pred))
print(metrics.mean_absolute_error(y_test, y_pred))

