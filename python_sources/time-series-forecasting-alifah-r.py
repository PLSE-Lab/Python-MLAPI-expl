#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (10,6)
pd.set_option('max_column', 100)
pd.set_option('max_row', 200)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


## Preprocess the data
def preprocess (dataset):
    processed_dataset = dataset.copy()
    processed_dataset['d-1']=processed_dataset['Revenue'].shift(periods=1)
    processed_dataset['d-2']=processed_dataset['Revenue'].shift(periods=2)
    processed_dataset['d-3']=processed_dataset['Revenue'].shift(periods=3)
    processed_dataset['d-4']=processed_dataset['Revenue'].shift(periods=4)
    return processed_dataset


# In[ ]:


## Split train and test
def split_train_test(dataset, end_of_training_date):
    
    
    return training_data, testing_data


# In[ ]:


## Split label and predictor
def split_label_and_predictor(train_or_test_data):
    
    # x_data = 
    # y_data = 
    
    return x_data, y_data


# In[ ]:


def fit(x_train, y_train):
    regr = RandomForestRegressor()
    regr.fit(x_train, y_train)
    return regr


# In[ ]:


def predict(est, x_test):
    y_pred = est.predict(x_test)
    return y_pred


# **Main Code**

# In[ ]:


df = pd.read_csv('/kaggle/input/uisummerschool/Online_sales.csv')
df['Date'] = df['Date'].astype(str)
df['Date'] = pd.to_datetime(df['Date'])
df.head()


# In[ ]:


df_rev = df.groupby('Date')['Revenue'].sum().reset_index(name='Revenue')
add_data = [['2017-12-01', 0], ['2017-12-02', 0], ['2017-12-03', 0],
            ['2017-12-04', 0], ['2017-12-05', 0], ['2017-12-06', 0],
            ['2017-12-07', 0], ['2017-12-08', 0], ['2017-12-09', 0],
            ['2017-12-10', 0], ['2017-12-11', 0], ['2017-12-10', 0],
            ['2017-12-13', 0], ['2017-12-14', 0]
           ] 
  
# Create the pandas DataFrame 
add_data_df = pd.DataFrame(add_data, columns = ['Date', 'Revenue']) 
add_data_df['Date'] = add_data_df['Date'].astype(str)
add_data_df['Date'] = pd.to_datetime(add_data_df['Date'])

df_rev = df_rev.append(add_data_df)
df_rev.head()


# In[ ]:


## Preprocess
daily_online_revenue = preprocess(df_rev).set_index('Date')
daily_online_revenue.head(10)


# In[ ]:


## Split
training_data, test_data = split_train_test(daily_online_revenue,"2017-11-30")
X_train, y_train = split_label_and_predictor(training_data)
X_test, y_test = split_label_and_predictor(test_data)


# In[ ]:


# Fit the model
model = fit(X_train, y_train)


# In[ ]:


# Predict the model
a = list(daily_online_revenue.iloc[-5:, :]['Revenue'])
for i in range(0, 5):
    temp = a[-5:]
    y = predict(model, [temp])
    a.append(y[0])


# In[ ]:


# Save the result to CSV

# Your code goes here

