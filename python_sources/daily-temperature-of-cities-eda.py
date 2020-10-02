#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:





# In[ ]:


temp=pd.read_csv("/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv")


# In[ ]:


temp.head()


# In[ ]:


temp['Region'].unique()


# In[ ]:


temp['Country'].unique()


# In[ ]:


temp['Month'].unique()


# In[ ]:


temp['Day'].unique()


# In[ ]:


temp['Year'].unique()


# In[ ]:


temp['AvgTemperature']


# In[ ]:


sns.distplot(temp['AvgTemperature'],kde=False)
#there's some extremely low temp, I'll just get rid of it. 


# In[ ]:


temp.head(3)


# In[ ]:


temp[temp['AvgTemperature'] < -80]


# In[ ]:


temp.drop(temp[temp['AvgTemperature']<-80].index,inplace=True)


# In[ ]:


sns.distplot(temp['AvgTemperature'],bins=100,kde=False)
#removed the outliers 


# In[ ]:


#city with the highest temperature 
temp[temp['AvgTemperature']==temp['AvgTemperature'].max()]


# In[ ]:


#city with the lowest temperature
temp[temp['AvgTemperature']==temp['AvgTemperature'].min()]


# In[ ]:


#graph of temperatures in different regions 


# In[ ]:


#grouping all the regions together 
regions=temp.groupby('Region')


# In[ ]:


regions.describe()


# In[ ]:


africa=temp[temp['Region']=='Africa']


# In[ ]:


sns.distplot(africa['AvgTemperature'],kde=False)


# In[ ]:


asia=temp[temp['Region']=='Asia']


# In[ ]:


sns.distplot(asia['AvgTemperature'],kde=False)


# In[ ]:


australia=temp[temp['Region']=='Australia/South Pacific']


# In[ ]:


sns.distplot(australia['AvgTemperature'],kde=False)


# In[ ]:


europe=temp[temp['Region']=='Europe']


# In[ ]:


sns.distplot(europe['AvgTemperature'],kde=False)


# In[ ]:


middleeast=temp[temp['Region']=='Middle East']


# In[ ]:


sns.distplot(middleeast['AvgTemperature'],kde=False,bins=100)


# In[ ]:


america=temp[temp['Region']=='North America']


# In[ ]:


sns.distplot(america['AvgTemperature'],kde=False,bins=100)


# In[ ]:


south_america=temp[temp['Region']=='South/Central America & Carribean']


# In[ ]:


sns.distplot(south_america['AvgTemperature'],kde=False,bins=100)


# In[ ]:


temp.head()


# In[ ]:


#fig=plt.figure()

#axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)


# In[ ]:


#we need to aggregate a date coloumn temp.head()


# In[ ]:


temp.drop('Date',axis=1,inplace=True)


# In[ ]:


temp.head()


# In[ ]:


import datetime


# In[ ]:


temp['Date']=pd.to_datetime((temp.Year*10000+temp.Month*100+temp.Day).apply(str),format='%Y%m%d')


# In[ ]:


temp.drop(['Month','Day','Year'],axis=1,inplace=True)
#The time coloumns have been converted to a single Date col


# In[ ]:


temp.head()


# In[ ]:


africa=temp[temp['Region']=='Africa']


# In[ ]:


africa['Year']=africa['Date'].map(lambda x:x.year)


# In[ ]:


africa.head()


# In[ ]:


af=africa.groupby('Year').mean().reset_index().drop(25)
af.head()


# In[ ]:


sns.lineplot(x='Year',y='AvgTemperature',data=af)


# In[ ]:


asia=temp[temp['Region']=='Asia']
asia['Year']=asia['Date'].map(lambda x:x.year)
asia1=asia.groupby('Year').mean().reset_index().drop(25)


# In[ ]:


asia1.head()


# In[ ]:


sns.lineplot(x='Year',y='AvgTemperature',data=asia1)


# In[ ]:


australia=temp[temp['Region']=='Australia/South Pacific']
australia['Year']=australia['Date'].map(lambda x:x.year)
aus=australia.groupby('Year').mean().reset_index().drop(25)


# In[ ]:


sns.lineplot(x='Year',y='AvgTemperature',data=aus)


# In[ ]:


europe=temp[temp['Region']=='Europe']
europe['Year']=europe['Date'].map(lambda x:x.year)
eu=europe.groupby('Year').mean().reset_index().drop(25)
sns.lineplot(x='Year',y='AvgTemperature',data=eu)


# In[ ]:


middleeast=temp[temp['Region']=='Middle East']
middleeast['Year']=middleeast['Date'].map(lambda x:x.year)
mid=middleeast.groupby('Year').mean().reset_index().drop(25)
sns.lineplot(x='Year',y='AvgTemperature',data=mid)


# In[ ]:


south_america=temp[temp['Region']=='South/Central America & Carribean']
south_america['Year']=south_america['Date'].map(lambda x:x.year)
sa=south_america.groupby('Year').mean().reset_index().drop(25)
sns.lineplot(x='Year',y='AvgTemperature',data=sa)


# In[ ]:


america=temp[temp['Region']=='North America']
america['Year']=america['Date'].map(lambda x:x.year)
am=america.groupby('Year').mean().reset_index().drop(25)
sns.lineplot(x='Year',y='AvgTemperature',data=am)


# In[ ]:




