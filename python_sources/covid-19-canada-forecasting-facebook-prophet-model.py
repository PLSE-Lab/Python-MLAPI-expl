#!/usr/bin/env python
# coding: utf-8

# # COVID-19 Canada Confirmed Cases and Fatalities Forecasting

# **In this notebook, the model will be predicting the cumulative number of confirmed COVID19 cases in Canada, as well as the number of resulting fatalities, for future dates. We understand this is a serious situation, and in no way want to trivialize the human impact this crisis is causing by predicting fatalities. Our goal is to provide better methods for estimates that can assist medical and governmental institutions to prepare and adjust as pandemics unfold. In this particular notebook popular facebook Prophet algorithm used.**

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


# **Loading Total Data**

# In[ ]:


train=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')
train.head()


# **Seperate Country Data**

# In[ ]:


country=train[(train.Country_Region=='Canada')]
country=country.groupby(country.Date).sum()


# In[ ]:


country['Date']=country.index
country.head()


# **Confirmed Cases Forecasting**

# In[ ]:


country_cc=country[['Date','ConfirmedCases']]
country_cc['ds']=country_cc['Date']
country_cc['y']=country_cc['ConfirmedCases']
country_cc.drop(columns=['Date','ConfirmedCases'], inplace=True)
country_cc.head()


# In[ ]:


from fbprophet import Prophet
model_cc=Prophet()
model_cc.fit(country_cc)


# In[ ]:


future = model_cc.make_future_dataframe(periods=100)
future.head()


# In[ ]:


forecast=model_cc.predict(future)
forecast.tail(5)


# In[ ]:


fig_Confirmed = model_cc.plot(forecast,xlabel = "Date",ylabel = "Confirmed")


# **Fatalities Forecasting**

# In[ ]:


country_ft=country[['Date','Fatalities']]
country_ft['ds']=country_ft['Date']
country_ft['y']=country_ft['Fatalities']
country_ft.drop(columns=['Date','Fatalities'], inplace=True)
country_ft.head()


# In[ ]:


from fbprophet import Prophet
model_ft=Prophet()
model_ft.fit(country_ft)


# In[ ]:


future = model_ft.make_future_dataframe(periods=100)
forecast=model_ft.predict(future)


# In[ ]:


fig_Fatalities = model_ft.plot(forecast,xlabel = "Date",ylabel = "Deaths")


# **#StayHome #StaySafe #May Almighty bless us All**

# **Please upvote if you like this or find this notebook useful, thanks.**
