#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.


# In[2]:


from fbprophet import Prophet


# In[3]:


def mean_abs_percentage_error(y_true,y_pred):
    y_true,y_pred = np.array(y_true),np.array(y_pred)
    return np.abs((y_true-y_pred)/y_true)


# In[14]:


df = pd.read_table('../input/random-load-data/data.txt',header=None)


# In[15]:


df.shape


# In[16]:


df.columns=['y']


# In[17]:


df.head()


# In[19]:


df['Date'] = pd.date_range(pd.datetime(2017,1,1),pd.datetime(2018,12,31))


# In[21]:


df.tail()


# In[22]:


df.head()


# In[23]:


data=df


# In[24]:


data.rename(columns={'Date':'ds'},inplace=True)


# In[33]:


from matplotlib.pylab import rcParams
#divide into train and validation set
train = data[(data.set_index('ds').index>=pd.datetime(2017,1,1)) & (data.set_index('ds').index<pd.datetime(2018,10,1))]
valid = data[data.set_index('ds').index>=pd.datetime(2018,10,1)]

rcParams['figure.figsize']=12,6

#plotting the data
plt.plot(train['ds'],train3['y'])
plt.plot(valid['ds'],valid3['y'])


# In[29]:


holidays = pd.read_excel('../input/the-national-archives-in-kew-uk/ukbankholidays.xls')


# In[30]:


holidays.head()


# In[31]:


holidays['holiday'] = 'national'
holidays['lower_window'] = 0
holidays['upper_window'] = 0
holidays.rename(columns={'UK BANK HOLIDAYS':'ds'},inplace=True)


# In[32]:


holidays.head()


# In[34]:


train.head()


# In[35]:


weather = pd.read_csv('../input/the-national-archives-in-kew-uk/KEW_WEATHER.csv')
weather['DATE'] = pd.to_datetime(weather['DATE'])


# In[36]:


temperature = weather[['DATE','TAVG']]
temperature.set_index('DATE',inplace=True)


# In[37]:


train_weather = temperature[(temperature.index>=pd.datetime(2016,1,1)) & (temperature.index<pd.datetime(2019,2,1))]


# In[38]:


test_weather =temperature[(temperature.index>=pd.datetime(2019,2,1)) & (temperature.index<=pd.datetime(2019,2,28))]


# In[39]:


dates=[]
values=[]
for i in train.set_index('ds').index:
    if(i not in train_weather.index):
        print(i)
        dates.append(pd.to_datetime(i))
        values.append(50+np.random.randint(-2,2))


# In[40]:


train_weather2 = pd.concat([train_weather,pd.DataFrame({'Date':dates,'TAVG':values}).set_index('Date')])


# In[41]:


for i in train_weather2.index:
    if(i not in train.set_index('ds').index):
        print(i)
        train_weather2.drop(index=i,inplace=True)


# In[42]:


train_weather2.shape


# In[43]:


train_weather3 = train_weather2[(train_weather2.index>=pd.datetime(2017,1,1)) & (train_weather2.index<pd.datetime(2019,2,1))]


# In[44]:


train2 = train.merge(train_weather2.reset_index(),left_on='ds',right_on='index').drop('index',axis=1)


# In[45]:


train3 = train3.merge(train_weather3.reset_index(),left_on='ds',right_on='index').drop('index',axis=1)


# In[ ]:


help(Prophet.add_regressor)


# In[46]:


model=Prophet()
# model.add_regressor('TAVG',prior_scale=5,standardize=False)
# model.add_country_holidays(country_name='UK')
# m.add_seasonality('self_define_cycle',period=8,fourier_order=8,mode='additive')


# In[47]:


model.fit(train)


# In[50]:


future = model.make_future_dataframe(periods=92)
future.tail()


# In[51]:


# Python
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[52]:


#plot the predictions for validation set
plt.plot(train.set_index('ds'), label='Train')
plt.plot(valid.set_index('ds'), label='Valid')
plt.plot(forecast[forecast['ds']>=pd.datetime(2018,10,1)].set_index('ds')['yhat'], label='Prediction',alpha=0.4)
# plt.legend()
# plt.save_fig('feb_forecast.png', bbox_inches='tight')


# In[53]:


fig2 = model.plot_components(forecast)


# In[55]:


fig, ax = plt.subplots()
plt.plot(valid.set_index('ds'), label='Valid',color='orange')
plt.plot(forecast[forecast['ds']>=pd.datetime(2018,10,1)].set_index('ds')['yhat'], label='Prediction',color='blue',alpha=0.4)
# upper=plt.plot(forecast[forecast['ds']>=pd.datetime(2019,2,1)].set_index('ds')['yhat_upper'], label='Prediction_upper',alpha=0.4,color='blue',)
# lower=plt.plot(forecast[forecast['ds']>=pd.datetime(2019,2,1)].set_index('ds')['yhat_lower'], label='Prediction_lower',alpha=0.4,color='blue')
# ax.fill_between(valid.set_index('ds').index,forecast[forecast['ds']>=pd.datetime(2019,2,1)].set_index('ds')['yhat_upper'],
#                 forecast[forecast['ds']>=pd.datetime(2019,2,1)].set_index('ds')['yhat_lower'], alpha=0.4)
plt.legend()
plt.savefig('three_years_train.png', bbox_inches='tight')


# In[ ]:


# forecast[forecast['ds']>=pd.datetime(2019,1,28)].set_index('ds')['yhat'].values
# valid.set_index('ds')


# In[56]:


#calculate rmse
from math import sqrt
from sklearn.metrics import mean_squared_error

rms = sqrt(mean_squared_error(valid.set_index('ds'),forecast[forecast['ds']>=pd.datetime(2018,10,1)].set_index('ds')['yhat']))
print(rms)


# In[59]:


np.mean(mean_abs_percentage_error(valid[valid['ds']>=pd.datetime(2018,10,1)].set_index('ds')['y'],forecast[forecast['ds']>=pd.datetime(2018,10,1)].set_index('ds')['yhat']))*100


# In[ ]:




