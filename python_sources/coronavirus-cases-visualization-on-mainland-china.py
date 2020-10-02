#!/usr/bin/env python
# coding: utf-8

# 
# **Coronavirus disease (COVID-19) outbreak**
# 
# (https://s7d2.scene7.com/is/image/TWCNews/0304_n13_covid_19_coronavirus_graphic_generic_file)
# 
# > The new coronavirus, now known as Covid-19, was first encountered in Wuhan, China, in December 2019, and has gone on to affect over 100,000 people in over 80 countries around the globe, causing more than 3,000 deaths.
# > The virus can cause pneumonia. Those who have fallen ill are reported to suffer coughs, fever and breathing difficulties. In severe cases there can be organ failure. As this is viral pneumonia, antibiotics are of no use. The antiviral drugs we have against flu will not work. If people are admitted to hospital, they may get support for their lungs and other organs, as well as fluids. Recovery will depend on the strength of their immune system. Many of those who have died were already in poor health.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly.graph_objects as go 

import seaborn as sns


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")
data.shape


# In[ ]:


data.head()


# In[ ]:


data1 = data.drop(['SNo', 'Last Update'], axis=1)


# In[ ]:


data1.head()


# In[ ]:


data1.info()


# **Create the List for New Data
# **
# > We create the new data as Observation Date in order to see the all numbers of Confirmed, Recovered and Deaths for all countries.

# In[ ]:


uni_dates = list(data1['ObservationDate'].unique())
confirmed=[]
recovered=[]
deaths=[]

for x in uni_dates:
    confirmed.append(data1[data1['ObservationDate']==x].Confirmed.sum())
    recovered.append(data1[data1['ObservationDate']==x].Recovered.sum())
    deaths.append(data1[data1['ObservationDate']==x].Deaths.sum())


line_new = pd.DataFrame()
line_new ['ObservationDate']=uni_dates
line_new['Confirmed']=confirmed
line_new['Recovered']=recovered
line_new['Deaths']=deaths
line_new.tail(10)


# > The new coronaviruses has spread from China's Hubei province to places around the globe.
# 
# > (https://i2.wp.com/metro.co.uk/wp-content/uploads/2020/01/virus_spread_6-bc11.gif?quality=90&strip=all&zoom=1&resize=644%2C322&ssl=1)

# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=line_new['ObservationDate'], 
                         y=line_new['Confirmed'],
                         mode='lines+markers',
                         name='Confirmed',
                         line=dict(color='Yellow', width=3)))
fig.add_trace(go.Scatter(x=line_new['ObservationDate'], 
                         y=line_new['Deaths'],
                         mode='lines+markers',
                         name='Deaths',
                         line=dict(color='Red', width=3)))
fig.add_trace(go.Scatter(x=line_new['ObservationDate'], 
                         y=line_new['Recovered'],
                         mode='lines+markers',
                         name='Recovered',
                         line=dict(color='Green', width=3)))

fig.show()


# In[ ]:


line_new = line_new.set_index('ObservationDate')
plt.style.use('default') 
plt.figure(figsize=(20,15))
sns.lineplot(data=line_new)
plt.xticks(rotation=15)
plt.title('Number of Coronavirus Cases Over Time', size =20)
plt.xlabel('Time', size=20)
plt.ylabel('Number of Cases', size=20)
plt.show()


# In[ ]:


plt.figure(figsize=(20,15))
line_new.Confirmed.plot(kind = 'line', color = 'm',label = 'Confirmed',linewidth=2,alpha = 1,linestyle = ':')
line_new.Deaths.plot(kind = 'line', color = 'k',label = 'Deaths',linewidth=3,alpha = 0.7,linestyle = '-.' )
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('Date')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()


# **Create the New Data as Mainland China Country**
# 
# > We only created the new data as mainland china country.Then we demonstrated the numbers of corona virus cases in all china by lineplot which it contains the confirmed ,recovered and deaths.

# In[ ]:


plt.figure(figsize=(20,15))
data3 = data1[data1['Country/Region'] =='Mainland China']
data3.Confirmed.plot(kind = 'line', color = 'y',label = 'Confirmed',linewidth=1,alpha = 1,linestyle = '--')
data3.Recovered.plot(kind = 'line',color = 'c',label = 'Recovered', linewidth = 3, linestyle = '-')
data3.Deaths.plot(kind = 'line', color = 'r',label = 'Deaths',linewidth=2,alpha = 0.7 )
plt.legend(loc='upper right')     # legend = puts label into plot
plt.ylabel('Values',fontsize= 16)
plt.title('Number of Corona Virus Cases in Mainland China',fontsize = 16)            # title = title of plot
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


x = np.array(data3.loc[:,'Confirmed']).reshape(-1,1)
y = np.array(data3.loc[:,'Recovered']).reshape(-1,1)
#Scatter
plt.figure(figsize=[10,10])
plt.scatter(x,y,color='Green')
plt.xlabel('Confirmed')
plt.ylabel('Recovered')
plt.title('Confirmed-Recovered in Mainland China')            # title = title of plot
plt.show()


# In[ ]:


x = np.array(data3.loc[:,'Confirmed']).reshape(-1,1)
y = np.array(data3.loc[:,'Deaths']).reshape(-1,1)
#Scatter
plt.figure(figsize=[10,10])
plt.plot(x,y,'-',lw=2, color='r')
plt.xlabel('Confirmed')
plt.ylabel('Deaths')
plt.title('Confirmed-Deaths in Mainland China')            # title = title of plot
plt.show()


# **Visualization of the Province/States in Mainland China
# **
# 
# We specified the Covid-19 viruses cases states in Mainland China. When we analizied these states according to the confirmed/recovery rate, the results were shown as follows.

# In[ ]:


f,ax1 = plt.subplots(figsize =(30,20))
sns.pointplot(x=data3['Province/State'],y=data3['Confirmed'],color = 'blue')
plt.xlabel("Province/States in Mainland China",fontsize = 16 , color = 'blue')
plt.ylabel("Confirmed Values",fontsize = 16 , color = 'blue')
plt.title("Confirmed Rate for Every Province/States in Mainland China",fontsize=20)


# In[ ]:


f,ax1 = plt.subplots(figsize =(30,20))
sns.pointplot(x=data3['Province/State'],y=data3['Deaths'],color = 'black')
plt.xlabel("Province/States in Mainland China",fontsize = 16 , color = 'red')
plt.ylabel("Deaths",fontsize = 16 , color = 'red')
plt.title("Death Rate for Every Province/States in Mainland China",fontsize=20)


# In[ ]:


f,ax1 = plt.subplots(figsize =(30,20))
sns.pointplot(x=data3['Province/State'],y=data3['Recovered'],color = 'green')
plt.xlabel("Province/States in Mainland China",fontsize = 16 , color = 'green')
plt.ylabel("Recovered",fontsize = 16 , color = 'green')
plt.title("Recovered Rate for Every Province/States in Mainland China",fontsize=20)


# > A total of 74,185 infections have been recorded in mainland China, most of them in Hubei province and its capital, Wuhan - the epicentre of the outbreak.
# > According to the previous chart values, you realize that the highest number of cases city is Hubei.We have shown the confirmed and recovered figures with barplot graphics in below.

# In[ ]:


data7 = data3[data3['Province/State'] =='Hubei']
data7.tail()


# In[ ]:



f,ax1 = plt.subplots(figsize =(40,30))
sns.barplot(x="Confirmed", y="Recovered",
                  hue="Province/State", data=data7)
plt.xlabel("Total Confirmed Numbers by Time",fontsize = 20 , color = 'green')
plt.ylabel("Total Recovered Numbers by Time",fontsize = 20 , color = 'green')
plt.show()


# In[ ]:


f,ax1 = plt.subplots(figsize =(35,20))
sns.barplot(x="Confirmed", y="Deaths",
                  hue="Province/State", data=data7,alpha=1,color='red')
plt.xlabel("Total Confirmed Numbers by Time",fontsize = 20 , color = 'red')
plt.ylabel("Total Deaths Numbers by Time",fontsize = 20 , color = 'red')
plt.show()

