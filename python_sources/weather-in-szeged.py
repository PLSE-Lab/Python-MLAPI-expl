#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **Weather in Szeged 2006-2016:** Is there a relationship between humidity and temperature? What about between humidity and apparent temperature? Can you predict the apparent temperature given the humidity?

# In[ ]:


weatherData = pd.read_csv('../input/szeged-weather/weatherHistory.csv',encoding='latin1')


# In[ ]:


for i in weatherData.columns:
    print(i)


# In[ ]:


apparenttemp = weatherData[['Apparent Temperature (C)']]
temperature = weatherData[['Temperature (C)']]
humidity = weatherData[['Humidity']]


# In[ ]:


humidity.isnull().sum()


# In[ ]:


plt.scatter(humidity, temperature, edgecolors='r')
plt.xlabel('humidity')
plt.ylabel('temperature')
plt.show()


# In[ ]:


weatherData['Temperature (C)'].corr(weatherData['Humidity'])


# In[ ]:


plt.scatter(humidity, apparenttemp, edgecolors='r')
plt.xlabel('humidity')
plt.ylabel('apparent temperature')
plt.show()


# In[ ]:


weatherData['Apparent Temperature (C)'].corr(weatherData['Humidity'])


# In[ ]:


from sklearn.model_selection  import train_test_split
x_train, x_test, y_train, y_test = train_test_split(humidity,apparenttemp,test_size=0.33,random_state=0)


# In[ ]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)


# In[ ]:


prediction = lr.predict(x_test)


# In[ ]:


plt.plot(x_train,y_train)
plt.plot(x_test,prediction)
plt.show()


# In[ ]:




