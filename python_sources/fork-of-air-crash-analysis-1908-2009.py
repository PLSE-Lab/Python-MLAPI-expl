#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


print(os.getcwd())


# In[ ]:


air_crash=pd.read_csv('../input/aircrash1908-2008.csv')


# In[ ]:


air_crash.head(2)


# In[ ]:


air_crash.tail(2)


# finding out that aircrash data we have is of 100 years from 1908 to 2009..

# In[ ]:


air_crash.describe()


# In[ ]:


air_crash.info()


# In[ ]:


import seaborn as sb
import matplotlib.pyplot as plt
from datetime import timedelta,date,datetime


# In[ ]:


#we have time stored in string format in the data table..so we create a function to convert the string format to date time format
def todate(x):
    return datetime.strptime(x,"%m/%d/%Y %H:%M")


# In[ ]:


#we now apply the function to the air_crash data
air_crash['Time']=air_crash['Time'].apply(todate)  #run this line once as the date time is converted to timestampand is not string after we run it once
print('Date n time ranges from  '+str(air_crash.Time.min())+'  to  '+str(air_crash.Time.max()))
air_crash.Operator=air_crash.Operator.str.upper()    # this is to avoid duplicate entries


# In[ ]:


#air_crash['Time']
#todate('09/17/1908 17:18')


# ## now we will find total accidents given the by year

# In[ ]:


#air_crash.Operator=air_crash.Operator.str.upper
# this is to avoid duplicate entries


# In[ ]:


accyear=air_crash.groupby(air_crash.Time.dt.year)[['Date']].count()
accyear.head()
accyear.columns=['Count']
accyear[accyear.index==1908]
#we get the count of accidents by the year


# In[ ]:


#plotting air crashes year wise
Temp=air_crash.groupby(air_crash.Time.dt.year)[['Date']].count() #temp is 
Temp=Temp.rename(columns={"Date":"Count"})

plt.figure(figsize=(12,6))
plt.style.use('bmh')
plt.plot(Temp.index,'Count',data=Temp,color='blue',marker='.',linewidth=1)
plt.xlabel('Year',fontsize=10)
plt.ylabel('Count',fontsize=10)
plt.title('Count of air accidents by Year',loc='Center',fontsize=14)
plt.show()


# In[ ]:


import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
plt.figure(figsize=(12,8))

plt.bar(air_crash.groupby(air_crash.Time.dt.month)[['Date']].count().index,'Date',data=air_crash.groupby(air_crash.Time.dt.month)[['Date']].count())
plt.xticks(air_crash.groupby(air_crash.Time.dt.month)[['Date']].count().index,['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])

plt.xlabel('Month',fontsize=10)
plt.ylabel('Count',fontsize=10)

plt.title('Count of air accidents by month',loc='Center',fontsize=14)
plt.show()


# In[ ]:


ax=pl.plot()

plt.bar(air_crash.groupby(air_crash.Time.dt.weekday)[['Date']].count().index,'Date',data=air_crash.groupby(air_crash.Time.dt.weekday)[['Date']].count(),color='lightskyblue',linewidth=2)
plt.xticks(air_crash.groupby(air_crash.Time.dt.weekday)[['Date']].count().index,['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])

plt.xlabel('Day of the week',fontsize=10)
plt.ylabel('Count',fontsize=10)

plt.title('Count of accidents by day of the week',loc='Center',fontsize=14)


# In[ ]:


ax=pl.plot()
plt.bar(air_crash[air_crash.Time.dt.hour != 0].groupby(air_crash.Time.dt.hour)[['Date']].count().index,'Date',data=air_crash[air_crash.Time.dt.hour !=0].groupby(air_crash.Time.dt.hour)[['Date']].count(),color='lightskyblue',linewidth=2)

plt.xlabel('Hour',fontsize=10)
plt.ylabel('Count',fontsize=10)
plt.title('Count of the air accidents by Hour',loc='Center',fontsize=14)

plt.show()


# now we check count of commercial or miltitary aircrafts in air crashes

# In[ ]:


tem=air_crash.copy()

tem['isMILITARY']=tem.Operator.str.contains('MILITARY')

tem=tem.groupby('isMILITARY')[['isMILITARY']].count()
tem.index=['Passenger','Military']
tem.head()


# In[ ]:


colors=['yellowgreen','lightskyblue']
plt.figure(figsize=(15,6))

plt.pie(tem.isMILITARY,colors=colors,labels=tem.isMILITARY,startangle=90)

plt.axis('equal')
plt.title('Total number of accidents by type of flight',loc='Center',fontsize=14)


# In[ ]:


#we are now counting individual air crashes for military and passenger by year and plotting the graphs
temp2=air_crash.copy()
temp2['Military']=temp2.Operator.str.contains('MILITARY')
temp2['Passenger']=temp2.Military==False
temp2=temp2.loc[:,['Time','Military','Passenger']]
temp2=temp2.groupby(temp2.Time.dt.year)[['Military','Passenger']].aggregate(np.count_nonzero)
temp2.head()


# In[ ]:


plt.figure(figsize=(15,10))
plt.plot(temp2.index,'Military',data=temp2,color='lightskyblue',marker='.',linewidth=1,label='military')
plt.plot(temp2.index,'Passenger',data=temp2,color='yellowgreen',marker='.',linewidth=1,label='passenger')
plt.xlabel('Year',fontsize=10)
plt.ylabel('Count',fontsize=10)
plt.title('Count of accidents by year',loc='Center',fontsize=14)
plt.legend(loc='upper right')
plt.show()


# In[ ]:


Fatalities=air_crash.groupby(air_crash.Time.dt.year).sum()
Fatalities.head()


# In[ ]:


Fatalities=air_crash.groupby(air_crash.Time.dt.year).sum()
Fatalities['Proportion']=Fatalities['Fatalities']/Fatalities ['Aboard']

plt.figure(figsize=(15,6))

plt.fill_between(Fatalities.index,'Aboard',data=Fatalities,color='skyblue')
plt.plot(Fatalities.index,'Aboard',data=Fatalities,marker='.',color='Slateblue',linewidth=1)
plt.fill_between(Fatalities.index,'Fatalities',data=Fatalities,color='olive',alpha=0.2)
plt.plot(Fatalities.index,'Fatalities',data=Fatalities,color='olive',marker='.',alpha=0.6,linewidth=1)

plt.legend(fontsize=10)
plt.xlabel('Year',fontsize=10)
plt.ylabel('Amount of people',fontsize=10)
plt.title('Total number of people involved by year',loc='Center',fontsize=14)
plt.show()


# In[ ]:




