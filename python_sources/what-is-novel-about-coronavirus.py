#!/usr/bin/env python
# coding: utf-8

# Coronavirus has become a big menance.It has been declared as a pandemic by world health organisation.Right now there are close to 2 lac cases reported and more than 7000 people have lost there lives.In India we have close to 150 cases reported.This virus has brought the world to a standstill and it will have grave implications on public health,economy and jobs.I have written a blog on Coronavirus which you can read from the link http://btplife.blogspot.com/2020/03/coronavirus.html . This kernel is a work in process I will be updating this in coming days.

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


# ### Importing the modules 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')


# ### Importing Data

# In[ ]:


df1 = pd.read_csv('../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv')
df2 = pd.read_csv('../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv')
df3 = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
df4 = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
df5 = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')
df6 = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')
df1.head(1)


# In[ ]:


df2.head(1)


# In[ ]:


df3.tail(1)


# In[ ]:


df4.head(1)


# In[ ]:


df5.head(1)


# In[ ]:


df6.head(1)


# ### Summary of DataSet

# In[ ]:


print('Rows     :',df3.shape[0])
print('Columns  :',df3.shape[1])
print('\nFeatures :\n     :',df3.columns.tolist())
print('\nMissing values    :',df3.isnull().values.sum())
print('\nUnique values :  \n',df3.nunique())


# ### Missing Values

# In[ ]:


total = df3.isnull().sum().sort_values(ascending=False)
percent = (df3.isnull().sum()/df3.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
f, ax = plt.subplots(figsize=(15, 6))
plt.xticks(rotation='90')
sns.barplot(x=missing_data.index, y=missing_data['Percent'])
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
missing_data.head()


# ### Renaiming Columns

# In[ ]:


df3 = df3.rename(columns={'Country/Region': 'Country', 'ObservationDate':'Date','Province/State':'State'})
df3.head(2)


# ### Basic Information

# In[ ]:


date = df3.groupby(["Date"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
date.head()


# Here We have summed up the Confirmed, Recovered and Death Numbers

# In[ ]:


print("Total Number of countries where Virus has Footprint:",len(df3["Country"].unique()))
print("Total Number of Confirmed Cases:",date["Confirmed"].iloc[-1])
print("Total Number of Recovered Cases:",date["Recovered"].iloc[-1])
print("Total Number of Death Cases:",date["Deaths"].iloc[-1])


# In[ ]:


plt.figure(figsize=(12,6))
plt.plot(date["Confirmed"],marker="o",label="Confirmed Cases")
plt.plot(date["Recovered"],marker="*",label="Recovered Cases")
plt.plot(date["Deaths"],marker="^",label="Death Cases")
plt.ylabel("Number of Patients")
plt.xlabel("Date")
plt.xticks(rotation=90)
plt.title("Coronavirus Spread Over Time")
plt.legend();


# Everyday the Pandemic is spreading like Wild Fire.The epicentre of Pandemic had moved from China to Europe and USA.In India we are in Stage 2 of Virus Spread with cases reported more than 300.

# ### Working with Datetime

# In[ ]:


df3['Date']=pd.to_datetime(df3['Date'])


# In[ ]:


#df['Hour']=df['timeStamp'].apply(lambda x:x.hour)
df3['Month']=df3['Date'].apply(lambda x:x.month)
df3['DayOfWeek']=df3['Date'].apply(lambda x:x.dayofweek)


# #### Cases per month

# In[ ]:


dmap={0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df3['DayOfWeek']=df3['DayOfWeek'].map(dmap)


# In[ ]:


mmap={1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
df3['Month']=df3['Month'].map(mmap)
#df3


# ### Details

# In[ ]:


# Sort data
df3 = df3.sort_values(['Date','Country','State'])
# Add column of days since first case
df3['first_date'] = df3.groupby('Country')['Date'].transform('min')
df3['days'] = (df3['Date'] - df3['first_date']).dt.days
#df3


# In[ ]:


latest = df3[df3.Date == df3.Date.max()]
#latest


# In[ ]:


cty = latest.groupby('Country').sum()
cty['Death Rate'] = cty['Deaths'] / cty['Confirmed'] * 100
cty['Recovery Rate'] = cty['Recovered'] / cty['Confirmed'] * 100
cty['Active'] = cty['Confirmed'] - cty['Deaths'] - cty['Recovered']
#cty.drop('days',axis=1).sort_values('Confirmed', ascending=False).head(10)


# In[ ]:


# fig, axes = plt.subplots(2, 2, figsize=(15, 10))
def plot_new(column, title):
    if column == 'Death Rate':
        _ = cty[cty.Deaths>=10].sort_values('Death Rate', ascending=False).head(15)
    else:
        _ = cty.sort_values(column, ascending=False).head(15)
    g = sns.barplot(_[column], _.index)
    plt.title(title, fontsize=14)
    plt.ylabel(None)
    plt.xlabel(None)
    plt.grid(axis='x')
    for i, v in enumerate(_[column]):
        if column == 'Death Rate':
            g.text(v*1.01, i+0.1, str(round(v,2)))
        else:
            g.text(v*1.01, i+0.1, str(int(v)))

plt.figure(figsize=(9,23))
plt.subplot(411)
plot_new('Confirmed','Confirmed cases top 15 countries')
plt.subplot(412)
plot_new('Deaths','Death cases top 15 countries')
plt.subplot(413)
plot_new('Active','Active cases top 15 countries')
plt.subplot(414)
plot_new('Death Rate','Death rate top 15 countries (>=10 deaths only)')
plt.show()


# Sadly we can see that the highest number of Deaths have occured in Italy.Straggle the countries with starting letter I Italy,Iran,Iraq and Indonesia have highest death rate.From Death rate we can see that the virus in more lethal in Middle East.Possible reason could be early detection and Healthcare facilities.

# ### Heatmap Based on Month

# In[ ]:


dayMonth=df3.groupby(by=['DayOfWeek','Month']).count()['SNo'].unstack()


# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(dayMonth,cmap='viridis');


# More cases get reported on Monday.Possibly this is because people avoid going to hospital on weekend.

# ### Cluster Map Based on Month

# In[ ]:


plt.figure(figsize=(12,6));
sns.clustermap(dayMonth,cmap='coolwarm');


# ### Forecasting the Future

# In[ ]:


confirmed = df3.groupby('Date').sum()['Confirmed'].reset_index()
deaths = df3.groupby('Date').sum()['Deaths'].reset_index()
recovered = df3.groupby('Date').sum()['Recovered'].reset_index()


# In[ ]:


confirmed.columns = ['ds','y']
#confirmed['ds'] = confirmed['ds'].dt.date
confirmed['ds'] = pd.to_datetime(confirmed['ds'])


# In[ ]:


confirmed.head()


# In[ ]:


from fbprophet import Prophet
m = Prophet(interval_width=0.95)
m.fit(confirmed)
future = m.make_future_dataframe(periods=7)
future_confirmed = future.copy() # for non-baseline predictions later on
future.tail()


# In[ ]:


forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[ ]:


confirmed_forecast_plot = m.plot(forecast)


# We can see that Our model is giving a lower prediction than actual confirmed cases.This is a case of concern the pandemic is spreading live wild fire and all countries are looking helpless in fron of it.

# In[ ]:


forecast_components = m.plot_components(forecast)


# We still have data for only three months.So its not sufficient to establish a Seasonal Trend.On weekely trend curve we can see that more cases are getting reported on Monday.
