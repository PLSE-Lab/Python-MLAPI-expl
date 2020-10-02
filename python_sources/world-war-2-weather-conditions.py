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


# [Weather Conditions in World War Two]: Is there a relationship between the daily minimum and maximum temperature? Can you predict the maximum temperature given the minimum temperature? 

# In[ ]:


weatherData = pd.read_csv('../input/weatherww2/Summary of Weather.csv',encoding='latin1')
# Shape of the dataframe.
weatherData.shape

# Data type of each column.
weatherData.dtypes

# Number of null values.
weatherData.info()


# To better understand the distribution of the variables Height and Weight, we can simply plot both variables using histograms. Histograms are plots that show the distribution of a numeric variable, grouping data into bins. The height of the bar represents the number of observations per bin.

# In[ ]:


plt.style.use('ggplot')

# Histogram of the Minimum Temperature
weatherData.MinTemp.plot(kind='hist',color='purple',edgecolor='black',figsize=(10,7))
plt.title('Distribution of Minimum Temperature', size=24)
plt.xlabel('Min Temp (C)', size=18)
plt.ylabel('Frequency', size=18)


# In[ ]:


# Histogram of the Maximum Temperature
weatherData.MaxTemp.plot(kind='hist',color='purple',edgecolor='black',figsize=(10,7))
plt.title('Distribution of Maximum Temperature', size=24)
plt.xlabel('Max Temp (C)', size=18)
plt.ylabel('Frequency', size=18);


# In[ ]:





# In[ ]:


dailyMin = weatherData[['MinTemp']]
dailyMax = weatherData[['MaxTemp']]


# In[ ]:


plt.scatter(dailyMin, dailyMax, edgecolors='r')
plt.xlabel('dailyMin')
plt.ylabel('dailyMax')
plt.show()


# In[ ]:


weatherData['MinTemp'].corr(weatherData['MaxTemp'], method = 'pearson')


# In[ ]:


from sklearn.model_selection  import train_test_split
x_train, x_test, y_train, y_test = train_test_split(dailyMin,dailyMax,test_size=0.33,random_state=0)


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




