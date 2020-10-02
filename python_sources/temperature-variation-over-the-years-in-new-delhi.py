#!/usr/bin/env python
# coding: utf-8

# This is the average temperature variation for the city of New Delhi, India from 1790 to 2013.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#reading the file and collecting only the timestamp and the year 
df= pd.read_csv('../input/GlobalLandTemperaturesByMajorCity.csv')
df_delhi = df[df['City']=='New Delhi']
df_delhi= df_delhi.ix[:, :2]
df_delhi


# In[ ]:


#selecting only the year from the time stamp and grouping the data by mean temperature of each year
a = df_delhi['dt'].apply(lambda x: int(x[0:4]))
grouped = df_delhi.groupby(a).mean()
grouped


# In[ ]:


#plotting the data
plt.plot(grouped['AverageTemperature'])
plt.show()
#As we can see there are several blank spaces due to the Nan's blocks in the data


# In[ ]:


#fixing the anomalies :-  by filling each Nan block in the original data by it's previous block's value 
#Then Plotting the fixed data
df_delhi['AverageTemperature'] = df_delhi['AverageTemperature'].fillna(method = 'ffill')
grouped = df_delhi.groupby(a).mean()
plt.plot(grouped['AverageTemperature'])
plt.xlabel('year')
plt.ylabel('temperature in degree celsius')
plt.title('New Delhi avreage temperature verus year')
plt.show()
#It can be seen that the average temperature is an increasing function for the most part


# In[ ]:


#modelling the data to obtain future values
from sklearn.linear_model import LinearRegression as LinReg


# In[ ]:


#reshaping the index of 'grouped'i.e years
x= grouped.index.values.reshape(-1,1)
#obtaining values of temperature
y = grouped['AverageTemperature'].values


# In[ ]:


#Using linear regression and finding accuracy of our prediction
reg = LinReg()
reg.fit(x,y)
y_preds = reg.predict(x)
Accuracy = str(reg.score(x,y))
print(Accuracy)


# In[ ]:


#plotting data along with regression
plt.scatter(x=x, y=y_preds)
plt.scatter(x=x,y=y, c='r')
plt.ylabel('Average Temperature in degree celsius')
plt.xlabel('year')
plt.show()


# In[ ]:


#finding future values of temperature
reg.predict(2048)


# In[ ]:


reg.predict(2020)

