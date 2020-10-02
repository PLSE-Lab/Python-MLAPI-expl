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


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression


# In[ ]:


plt.style.use('fivethirtyeight')


# In[ ]:


pas = pd.read_csv('/kaggle/input/russian-passenger-air-service-20072020/russian_passenger_air_service_2.csv')
package = pd.read_csv('/kaggle/input/russian-passenger-air-service-20072020/russian_air_service_CARGO_AND_PARCELS.csv')


# In[ ]:


pas.head()


# In[ ]:


pas.isna().sum()/len(pas)*100


# In[ ]:


pas.head()


# In[ ]:


long = pas.melt(id_vars = ['Airport name', 'Year'],
         value_vars = ['January', 'February', 'March', 'April', 'May','June', 'July', 'August', 'September', 'October', 'November','December'])

long.rename(columns = {'variable':'month', 'value':'passengers'}, inplace = True)


# In[ ]:


# sort months in right order

months = ['January', 'February', 'March', 'April', 'May','June', 'July', 'August', 'September', 'October', 'November','December']
long['month'] = pd.Categorical(long['month'], categories=months, ordered=True)


# In[ ]:


long


# In[ ]:


per_year = long.groupby('Year')['passengers'].sum()/1000000
per_year = per_year.reset_index()


# In[ ]:


plt.figure(figsize=(20,11))

sns.barplot(data = per_year, x = 'Year', y = 'passengers')
plt.xlabel('Year', fontsize = 16)
plt.ylabel('Passengers in millions', fontsize = 16)
plt.title("Passengers per Year in Russia's airports", fontsize = 24)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


# ## Time Series of Russias's airline passengers from 2007-2020

# In[ ]:


ts =  long.groupby(['Year', 'month'])['passengers'].sum()


# In[ ]:


ts.to_frame()


# In[ ]:


ts = ts.reset_index()

time = range(1,169)
ts['time'] = time                            # create time variable
ts['pasmio'] = ts['passengers'] / 1000000    # change representation to millions


# In[ ]:


plt.figure(figsize = (20,10))

sns.scatterplot(data = ts, x = 'time', y = 'pasmio', color = 'black')
sns.lineplot(data = ts, x = 'time', y = 'pasmio', color = 'r')
plt.ylabel('Passengers in million', fontsize=  16)
plt.title("Time Series of Russias's airline passengers from 2007-2020", fontsize = 19)


# ## Clear trend and seasonality is visible
# 
# ### Peak in July, low in December
# ### Constant upward trend
# 

# In[ ]:


ts


# In[ ]:


from statsmodels.tsa.stattools import adfuller
results = adfuller(ts['pasmio'])
print(results[1])


# In[ ]:


ts['log_pas'] = np.log(ts['pasmio'])


# In[ ]:


X = ts['time'].values
X = np.reshape(X, (len(X), 1))


# In[ ]:


y = ts['pasmio'].values


# In[ ]:


model = LinearRegression()
model.fit(X, y)


# In[ ]:


trend = model.predict(X)


# In[ ]:


plt.figure(figsize = (20,10))

sns.lineplot(data = trend)
sns.lineplot(data = ts, x = 'time', y = 'pasmio', color = 'r')
plt.ylabel('Passengers in million', fontsize=  16)
plt.title("Time Series of Russias's airline passengers from 2007-2020", fontsize = 19)


# In[ ]:


model


# Removing linear trend 

# In[ ]:


detrended = [y[i]-trend[i] for i in range(0, len(ts))]


# In[ ]:


plt.figure(figsize = (20,10))

plt.plot(detrended)
plt.show()


# We can still clearly see the problem of heteroscedasticity.
# 
# 
# Differenced time series

# In[ ]:


plt.figure(figsize = (20,10))

X = ts['pasmio'].values

diff = list()
for i in range(1, len(X)):
	value = X[i] - X[i - 1]
	diff.append(value)
plt.plot(diff)
plt.show()


# In[ ]:


from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf


# In[ ]:


detrended = pd.Series(detrended)


# In[ ]:


plot_acf(ts['pasmio'])


# In[ ]:


plot_acf(detrended)


# We can also still see the yearly periodic pattern

# In[ ]:




