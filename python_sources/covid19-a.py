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


import numpy as np
import pandas as pd


# In[ ]:


train_set = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_train.csv')


# In[ ]:


train_set.tail()


# In[ ]:


# Drop row with 0 confirmed cases, i.e., before tracking the virus infection
train_set = train_set[train_set['ConfirmedCases'] != 0]
# Drop columns that do not provide any useful informatio
train_set.drop(['Province/State', 'Country/Region','Lat', 'Long'], axis=1, inplace=True)


# In[ ]:


# Take a look at the incremental percentage changes
train_set['Fatalities_diff'] = train_set['Fatalities'].pct_change() * 100
train_set['Confirmed_diff'] = train_set['ConfirmedCases'].pct_change() * 100


# In[ ]:


# Elementary Data Analysis
# Some basic statistics of the data set
train_set.describe()


# In[ ]:


#set up environment for basic visualization
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


# In[ ]:


f = plt.figure(figsize=(8,10))
# plot percentage change
ax = f.add_subplot(211) 
ax.plot(train_set['Id'], train_set['Confirmed_diff'])
ax.plot(train_set['Id'], train_set['Fatalities_diff'])
ax.set_xlabel('Id')
ax.set_ylabel('Pecentage Change')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())

# plot actual cases
ax2 = f.add_subplot(212) 
l1 = ax2.plot(train_set['Id'], train_set['ConfirmedCases'], label='Confirmed Cases')
ax2.set_xlabel('Id')
ax2.set_ylabel('Number of Confirmed Cases')
#ax2.legend(loc=0)
color='red'
ax2b = ax2.twinx()
l2 =ax2b.plot(train_set['Id'], train_set['Fatalities'], color=color, label='Fatalitoes')
ax2b.set_ylabel('Number of Death')
# combine all legend into one box
ls = l1 + l2
labs = [l.get_label() for l in ls]
ax2.legend(ls, labs, loc=0)


# In[ ]:


# visualize the relationship between Confirmed cases and death tolls
plt.plot(train_set.ConfirmedCases, train_set.Fatalities)
#plt.plot(train_set.Id, d_label, label='actual')
#plt.legend()
plt.xlabel('Confirmed Cases')
plt.ylabel('Number of Deaths')
plt.title('Number of Deaths/Confirmed Case')
plt.show()


# In[ ]:


# Prepare data for linear regression
# Assume a quadratic form
train_set['Id2'] = train_set['Id'] ** 2

# Input for death toll prediction
d_train = train_set[['Id', 'Id2', 'ConfirmedCases']]

# Input for confirmed cases prediction
cf_train = train_set[['Id', 'Id2']]


# In[ ]:


# Dependent variable
cf_label = train_set['ConfirmedCases']
d_label = train_set['Fatalities']


# A natural log based regression

# In[ ]:


lg_cf_label = np.log(cf_label)
lg_d_label = np.log(d_label)


# In[ ]:


lg_train = train_set[['Id']]


# In[ ]:


plt.plot(lg_train, lg_cf_label, label='actual')
plt.legend()
plt.title('Log Number of Confirmed Cases')
plt.show()


# In[ ]:


plt.plot(lg_train, lg_d_label, label='actual')
plt.legend()
plt.title('Log Number of Death Tolls')
plt.show()


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[ ]:


#regression for confirmed case
lg_cf_reg = LinearRegression().fit(lg_train, lg_cf_label)
lg_cf_reg.score(lg_train, lg_cf_label)


# In[ ]:


lg_cf_pred = lg_cf_reg.predict(lg_train)
lg_cf_mse = mean_squared_error(lg_cf_pred, lg_cf_label)
lg_cf_mse


# In[ ]:


plt.plot(lg_train, np.transpose(lg_cf_pred), label='prediction')
plt.plot(lg_train, lg_cf_label, label='actual')
plt.legend()
plt.title('Log Confirmed Cases')
plt.show()


# In[ ]:


ppred = np.exp(lg_cf_pred)


# In[ ]:


plt.plot(lg_train, np.transpose(np.exp(lg_cf_pred)), label='prediction')
plt.plot(lg_train, cf_label, label='actual')
plt.legend()
plt.title('Confirmed Cases')
plt.show()


# In[ ]:


#regression for death tolls
lg_d_reg = LinearRegression().fit(lg_train, lg_d_label)
lg_d_reg.score(lg_train, lg_d_label)


# In[ ]:


lg_d_pred = lg_d_reg.predict(lg_train)
lg_d_mse = mean_squared_error(lg_d_pred, lg_d_label)
lg_d_mse


# In[ ]:


plt.plot(lg_train, np.transpose(lg_d_pred), label='prediction')
plt.plot(lg_train, lg_d_label, label='actual')
plt.legend()
plt.title('log death tolls')
plt.show()


# In[ ]:


plt.plot(lg_train, np.transpose(np.exp(lg_d_pred)), label='prediction')
plt.plot(lg_train, d_label, label='actual')
plt.legend()
plt.title('death tolls')
plt.show()


# Now make some forecasts...

# In[ ]:


test_set = pd.read_csv('/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_test.csv')


# In[ ]:


test_x = test_set[['ForecastId']] + 50


# In[ ]:


cf_forecast = np.exp(lg_cf_reg.predict(test_x))
d_forecast = np.exp(lg_d_reg.predict(test_x))


# In[ ]:


forecast = test_x.copy()
forecast['ForecastId'] = forecast['ForecastId'] - 50
forecast['ConfirmedCases'] = cf_forecast
forecast['Fatalities'] = d_forecast


# In[ ]:


forecast.to_csv('/kaggle/working/submission.csv', index=False)

