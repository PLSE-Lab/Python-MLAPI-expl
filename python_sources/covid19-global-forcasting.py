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


# Import appropriate libraries 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from fbprophet import Prophet
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Read all the CSV files and create dataframes
df = pd.read_csv('../input/covid19-global-forecasting-week-3/train.csv') # from Kaggle
df1 = pd.read_csv('../input/covid19-global-forecasting-week-3/test.csv')
df2 = pd.read_csv('../input/covid19-global-forecasting-week-3/submission.csv')

df.head()


# In[ ]:


df2.plot()


# In[ ]:


df.plot()


# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(df.isnull(), cbar = False, cmap = 'YlGnBu')


# In[ ]:


#setting the index to be the last_update

df.index = pd.DatetimeIndex(df.Date)


# In[ ]:


df


# In[ ]:


df['Country_Region'].value_counts()


# In[ ]:


df['Province_State'].value_counts()


# In[ ]:


# Let's plot and see the status

plt.figure(figsize = (15,10))
sns.countplot(y = 'Country_Region', data = df, order = df['Country_Region'].value_counts().iloc[:15].index)


# In[ ]:


df.resample('Y').size()


# In[ ]:


df.resample('M').size()


# In[ ]:


# Let's see the frequency

plt.plot(df.resample('M').size())
plt.title('Country Wise Per Month')
plt.xlabel('Months')
plt.ylabel('Number of confirmed cases')


# In[ ]:


# Preparing the data for forcast

df_prophet = df.resample('M').size().reset_index()


# In[ ]:


df_prophet


# In[ ]:


df_prophet.columns = ['Date', 'Confirmed_Cases']


# In[ ]:


df_prophet


# In[ ]:


df_prophet_df = pd.DataFrame(df_prophet)


# In[ ]:


df_prophet_df


# In[ ]:


# Make Predictions

df_prophet_df.columns


# In[ ]:


df_prophet_final = df_prophet_df.rename(columns={'Date': 'ds', 'Confirmed_Cases': 'y'})


# In[ ]:


df_prophet_final


# In[ ]:


m = Prophet()
m.fit(df_prophet_final)


# In[ ]:


# Forecasting future cases for 90 days

future = m.make_future_dataframe(periods = 90)
forecast = m.predict(future)


# In[ ]:


forecast


# In[ ]:


# Let's see explore forecast for next 90 days

figure = m.plot(forecast, xlabel='Date', ylabel = 'Confirmed_cases')


# In[ ]:


figure2 = m.plot_components(forecast)


# In[ ]:




