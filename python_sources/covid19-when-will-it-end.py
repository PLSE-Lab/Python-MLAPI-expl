#!/usr/bin/env python
# coding: utf-8

# Find out how long for this COVID-19 epidemic
# 
# Factors that impact to number of COVID infected cases , Recovered Rate , Deathe Rate

# If you found any error , please run import cell below again. And go back to cell you found error . Run again , It's work.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import math

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from datetime import datetime

import folium
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster, TimestampedGeoJson
# Any results you write to the current directory are saved as output.


# Compare withSAR 2003

# Data period during 2003-01-17 till 2003-07-11 total 96 Days.
# 
# But from this link SAR may start since Nov-2002.
# 
# https://en.wikipedia.org/wiki/2002%E2%80%9304_SARS_outbreak

# In[ ]:


sar=pd.read_csv('/kaggle/input/sars-outbreak-2003-complete-dataset/sars_2003_complete_dataset_clean.csv')


# In[ ]:


sar['Remain']=sar['Cumulative number of case(s)']-sar['Number recovered']-sar['Number of deaths']


# In[ ]:


top_n=40
top_sar_country=sar.groupby(['Country']).agg({'Cumulative number of case(s)':'max'}).reset_index().sort_values(by=['Cumulative number of case(s)'],ascending=False).head(top_n)


# In[ ]:


top_sar_country.head()


# In[ ]:


plt.figure(figsize=(16, 6))
plt.xticks(rotation=90)
plt.bar(x=top_sar_country['Country'],height=top_sar_country['Cumulative number of case(s)'])
plt.show()


# In[ ]:


top_sar_country[top_sar_country['Cumulative number of case(s)']>100]


# Coutries with Number of SAR case more than 100 is only 6 countries. 

# In[ ]:


plt.figure(figsize=(16, 6))
plt.xticks(rotation=90)
china_sar=sar[(sar['Country']=='China') & (sar['Cumulative number of case(s)']>0)]
x=china_sar['Date'].values
plt.plot(sar[(sar['Country']=='China') & (sar['Cumulative number of case(s)']>0)]['Date'],sar[(sar['Country']=='China') & (sar['Cumulative number of case(s)']>0)]['Remain'])
plt.axvline(x='2003-05-13',ymin= 0, label='pyplot vertical line',color='r')
plt.xlabel('Days since first case')
plt.ylabel('Remain Infected cases')
plt.annotate('60 days from peak infected at 3068 case till 38 infected case at 2003-07-11', (41,3100))
plt.xticks(x[::2])
plt.show()


# Remain = Confirmed Cases - Recovered Cases - Death Cases.
# For China, we can count days from Peak to Remain cases = 38 is **60** Days.(Due to end of data)

# *Difficult to count from  1st day because data is not complete. SAR 2003 started spread since Nov-2002.https://en.wikipedia.org/wiki/2002%E2%80%932004_SARS_outbreak

# In[ ]:


plt.figure(figsize=(16, 6))
plt.xticks(rotation=90)
start_top=0
for c in top_sar_country['Country'].values[start_top:]:
    plt.plot(sar[(sar['Country']==c) & (sar['Cumulative number of case(s)']>0)]['Remain'].values)
plt.gca().legend(top_sar_country['Country'].values[start_top:start_top+10])
plt.xlabel('Days since first case')
plt.ylabel('Remain Infected cases')
plt.show()


# Peak time different in each country . Assume that total days is 96 Days.

# In[ ]:


sar['Date'].nunique()


# In[ ]:


plt.figure(figsize=(16, 12))
plt.xticks(rotation=90)
start_top=3
for c in top_sar_country['Country'].values[start_top:]:
    plt.plot(sar[(sar['Country']==c) & (sar['Cumulative number of case(s)']>0)]['Remain'].values)
plt.gca().legend(top_sar_country['Country'].values[start_top:start_top+20])
plt.xlabel('Days since first case')
plt.ylabel('Remain Infected cases')
plt.show()


# Start from Top 4th. Number of remain is decrease but in some country eg. Canada ,remain cases increase agian make 2 peaks.  

# In[ ]:


x=['China','Other countries']
y=[top_sar_country[top_sar_country['Country']=='China']['Cumulative number of case(s)'].values[0],np.sum(top_sar_country[top_sar_country['Country']!='China']['Cumulative number of case(s)'])]

plt.bar(x,y)
for i, v in enumerate(y):
    plt.annotate(str(v), ( x[i],y[i]))
plt.show()


# In[ ]:


plt.pie(y,labels=['China','Other Countries'],autopct='%1.1f%%')
plt.show()


# Total SAR Case =  8645 Confirmed.China 5329 , Out of China 3316. 61.6% is in China.
# 

# Compare with case out of china of SAR on 2003 and multiply by increasing number of outbound travel .Exclude China ,COVID estimation is may reach 796k.
# (Max confirmed case in China is 80k) 
# 
# Anyway , Chinese outbound travel data on 2003 is about 5M , but 2018 increase to 80M.
# Travel Data : https://china-outbound.com/editorial-latest-chinese-data-h1-2018/
# 
# The 796k infected all over the world is the maximum number cause all coutries have a lot awareness to fight with covid.

# In[ ]:


# Calcualte Total case of COVID by using historical SAR data and growth in China outbound tarveller  
81054*3316/5329*80/5


# > **Let's analyze COVID data**

# In[ ]:


covid_pd=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')


# In[ ]:


covid_pd.tail()


# In[ ]:


covid_pd.rename(columns={'Country/Region':'Country'},inplace=True)
covid_pd=covid_pd.replace(to_replace='Mainland China',value='China')
covid_pd=covid_pd.replace(to_replace=' Azerbaijan',value='Azerbaijan')
covid_pd=covid_pd.replace(to_replace='UK',value='United Kingdom')
covid_pd=covid_pd.replace(to_replace='US',value='United States')


# In[ ]:


covid_pd['Remain']=covid_pd['Confirmed']-covid_pd['Recovered']-covid_pd['Deaths']


# Analyze data for China Province

# In[ ]:


lat_long=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')


# In[ ]:


lat_long.head()


# In[ ]:


china_province_covid=covid_pd[covid_pd['Country']=='China'].groupby('Province/State').agg({'Confirmed':'max'}).reset_index()


# In[ ]:


china_lat_long=pd.merge(china_province_covid,lat_long,on='Province/State')


# In[ ]:


cn_geo_data = "../input/china-regions-map/china.json"


# In[ ]:


#I use code from this link. His map is so cool.
#https://www.kaggle.com/pongsaksawang/coronavirus-propagation-visualization-forecast

map = folium.Map(location=[35, 105], 
                 tiles = "CartoDB dark_matter",
                 detect_retina = True,
                 zoom_start=4)

tooltip = 'Hubei'
lat=china_lat_long[china_lat_long['Province/State']=='Hubei']['Lat']
lon=china_lat_long[china_lat_long['Province/State']=='Hubei']['Long']
con=str(int(china_lat_long[china_lat_long['Province/State']=='Hubei']['Confirmed'].values[0]))
folium.Marker([lat, lon], popup=con+" cases", tooltip=tooltip).add_to(map)

folium.Choropleth(
    geo_data=cn_geo_data,
    name='choropleth',
    key_on='feature.properties.name',
    fill_color='blue',
    fill_opacity=0.18,
    line_opacity=0.7
).add_to(map)

for i in range(len(china_lat_long)):
      folium.CircleMarker(location = [china_lat_long.loc[i,'Lat'],china_lat_long.loc[i,'Long']], 
                        radius = np.log(china_lat_long.loc[i,'Confirmed'])*3, 
                    
                        color = '#E80018', 
                        fill_opacity = 0.7,
                        weight = 2, 
                        fill = True
                       ,fillColor = '#E80018'
                         ).add_to(map)

map


# ![](https://www.travelchinaguide.com/images/photogallery/2018/january-weather.jpg)

# I cannot find temperature by city data. For quick look , I got temperature map picture from https://www.travelchinaguide.com/tour/weather-in-january.htm (Spreading Period in China is in Jan-20 )
# 
# Colder Western region have lower cases compare to eastearn with higher temperature 0-10 Celsius.But this may be western is less density of population.

# In[ ]:


country_covid=covid_pd.groupby(['ObservationDate','Country']).agg({'Confirmed':'max','Deaths':'max','Recovered':'max','Remain':'sum'}).reset_index()


# In[ ]:


top_n=10
top_country=covid_pd.groupby(['Country']).agg({'Remain':'max'}).reset_index().sort_values(by=['Remain'],ascending=False).head(top_n)


# In[ ]:


country_covid_remain=country_covid.pivot(index='ObservationDate',columns='Country',values='Remain').reset_index()
country_covid_remain=country_covid_remain.fillna(0)


# In[ ]:


plt.figure(figsize=(16, 6))
plt.xticks(rotation=90)
start_top=0
df=country_covid_remain[top_country['Country'].values[start_top]]
plt.bar(country_covid_remain['ObservationDate'],df)
for c in top_country['Country'].values[start_top+1:]:
    plt.bar(country_covid_remain['ObservationDate'],country_covid_remain[c],bottom=df)
    df=df+country_covid_remain[c]

plt.gca().legend(top_country['Country'].values[start_top:start_top+top_n])
plt.xlabel('Date')
plt.ylabel('Remain Infected cases')
#plt.yscale("log")
plt.show()


# Number of remain covid cases is decreasing, but for other coutries like Italy,South Korea, and Iran are increasing. So remain case will be increasing again. 
# 
# For covid , the second peak may be higher than 1st peak (by China) because number of outbond traveler from China is lot higher than SAR spread in 2003 and occur during chinese new year which is travel time for chinese.  

# In[ ]:


top_n=10
top_country=covid_pd.groupby(['Country']).agg({'Remain':'max'}).reset_index().sort_values(by=['Remain'],ascending=False).head(top_n)


# In[ ]:


plt.figure(figsize=(16, 8))
plt.xticks(rotation=90)
top_c=np.append(top_country['Country'].values,"South Korea")
start_top=0
for c in top_c:
    plt.plot(country_covid[(country_covid['Country']==c) & (country_covid['Remain']>0)]['Remain'].values)
plt.plot(country_covid[(country_covid['Country']=='South Korea') & (country_covid['Remain']>0)]['Remain'].values)
plt.gca().legend(top_c)
plt.xlabel('Days since first case')
plt.ylabel('Remain Infected cases')

plt.show()


# From above, we can see spreading time within country.Some country eg. Iran reach 1000 cases in 12 days. For South korea take more than 30 days to reach 1000 and Italy a bit faster than South Korea for 3-4 Days.   

# In[ ]:


top_n=5
top_country=covid_pd.groupby(['Country']).agg({'Remain':'max'}).reset_index().sort_values(by=['Remain'],ascending=False).head(top_n)


# In[ ]:


#To compare with China need log(data)

plt.figure(figsize=(16, 8))
plt.xticks(rotation=90)

#Add 30 days for china casue spread start since end of Dec-19
china_p=np.ones(66)
a=country_covid[(country_covid['Country']=='China') & (country_covid['Remain']>0)]['Remain'].values
china_p=np.append(china_p,a)
plt.plot(china_p)

top_c=np.append(top_country['Country'].values,["South Korea","Switzerland",'Iran','Thailand'])
start_top=0
for c in top_c:
    plt.plot(country_covid[(country_covid['Country']==c) & (country_covid['Remain']>0)]['Remain'].values)
plt.gca().legend(np.append("China",top_c))
plt.axvline(x=92,ymin= 0, label='pyplot vertical line',color='r')
plt.axvline(x=77,ymin= 0, label='pyplot vertical line',color='b')
plt.axvline(x=53,ymin= 0, label='pyplot vertical line',color='c')
plt.axvline(x=45,ymin= 0, label='pyplot vertical line',color='y')
plt.axvline(x=35,ymin= 0, label='pyplot vertical line',color='r')
plt.xlabel('Days since first case')
plt.ylabel('Remain Infected cases')
plt.yscale("log")
plt.annotate('China Peak at 92 Days', (92,2000))
plt.annotate('Thailand Peak at 77 Days', (77,1000))
plt.annotate('South Korea Peak at 53 Days', (53,300))
plt.annotate('Iran Peak at 46 Days', (45,100))
plt.annotate('Switzerland Peak at 35 Days', (35,20))

plt.show()


# Chart of Log(Confirmed) vs Day since first case
# 
# Compare to China the peak is on 55th Day (add 30 days casue spread start end of 17-Nov-19)
# 
# For China , slope decline obviously . When compare to SAR which take 2 Month before back to normal, China will be back to normal on mid of Apr-20.
# 

# In[ ]:


country='South Korea'
peak_remain=np.max(country_covid[(country_covid['Country']==country)]['Remain'])
peak_d=country_covid[(country_covid['Country']==country) & (country_covid['Remain']==peak_remain)]['ObservationDate'].values[0]
st_d=country_covid[(country_covid['Country']==country)]['ObservationDate'].values[0]
print(peak_d,datetime.strptime(peak_d,'%m/%d/%Y')-datetime.strptime(st_d,'%m/%d/%Y'))


# In[ ]:


top_n=100
top_country=covid_pd.groupby(['Country']).agg({'Remain':'max'}).reset_index().sort_values(by=['Remain'],ascending=False).head(top_n)
current_date=max(covid_pd['ObservationDate'])
peak_ca=[]
peak_da=[]
peak_date=[]
for c in  top_country['Country'].values:
    peak_remain=np.max(country_covid[(country_covid['Country']==c)]['Remain'])
    peak_d=country_covid[(country_covid['Country']==c) & (country_covid['Remain']==peak_remain)]['ObservationDate'].values[0]
    st_d=country_covid[(country_covid['Country']==c)]['ObservationDate'].values[0]
    if ((datetime.strptime(current_date,'%m/%d/%Y')-datetime.strptime(peak_d,'%m/%d/%Y')).days)>=5:
        peak_ca.append(c)
        peak_da.append((datetime.strptime(peak_d,'%m/%d/%Y')-datetime.strptime(st_d,'%m/%d/%Y')).days)
        peak_date.append(peak_d)
        #print(c,peak_d,(datetime.strptime(peak_d,'%m/%d/%Y')-datetime.strptime(st_d,'%m/%d/%Y')).days)
        
peak_da=np.reshape(peak_da,(-1,1))
peak_ca=np.reshape(peak_ca,(-1,1))
peak_date=np.reshape(peak_date,(-1,1))
r=np.concatenate((peak_ca,peak_da,peak_date),axis=1)
country_peak=pd.DataFrame(data=r,columns=['Country','Peak Day','Peak Date'])
country_peak['Peak Day']=country_peak['Peak Day'].astype('int')
country_peak=country_peak[(country_peak['Country']!='China')]
country_peak=country_peak[(country_peak['Country']!='Others')]
country_peak=country_peak.sort_values(by=['Peak Day'])
country_peak


# In[ ]:


plt.figure(figsize=(16, 8))
plt.bar(country_peak['Country'],height=country_peak['Peak Day'])
plt.xticks(rotation=90)
plt.show()


# Above Chart , Show Country that pass peak day . Y axis show number before peak since first case. Show how fast they can control.

# In[ ]:


country_covid['ObservationDate']=pd.to_datetime(country_covid['ObservationDate'])


# In[ ]:


select=['Italy','United States','Germany','France','Spain']
plt.xticks(rotation=90)
for c in select:
     pd=country_covid[country_covid['Country']==c].sort_values(by=['ObservationDate'])
     plt.plot(pd['ObservationDate'].values,pd['Remain'])
plt.gca().legend(select)
plt.yscale("log")
plt.show()


# In[ ]:


select=['South Korea','Iran']
plt.xticks(rotation=90)
for c in select:
     pd=country_covid[country_covid['Country']==c].sort_values(by=['ObservationDate'])
     plt.plot(pd['ObservationDate'].values,pd['Remain'])
plt.gca().legend(select)
plt.yscale("log")
plt.show()


# In[ ]:


select=['Switzerland','Netherlands','United Kingdom','Sweden','Belgium']
plt.xticks(rotation=90)

for c in select:
     pd=country_covid[country_covid['Country']==c].sort_values(by=['ObservationDate'])
     plt.plot(pd['ObservationDate'].values,pd['Remain'])
plt.gca().legend(select)
plt.yscale("log")
plt.show()


# In[ ]:


select=['Taiwan','Hong Kong','Singapore','Malaysia','Thailand','Japan']

plt.xticks(rotation=90)
for c in select:
     pd=country_covid[country_covid['Country']==c].sort_values(by=['ObservationDate'])
     plt.plot(pd['ObservationDate'].values,pd['Confirmed'])
plt.gca().legend(select)
plt.yscale("log")
plt.show()


# Confirmed Case

# In[ ]:


select=['Taiwan','Hong Kong','Thailand','Malaysia','Singapore','Japan']
plt.xticks(rotation=90)
for c in select:
     pd=country_covid[country_covid['Country']==c].sort_values(by=['ObservationDate'])
     plt.plot(pd['ObservationDate'].values,pd['Remain'])
plt.gca().legend(select)
plt.yscale("log")
plt.show()


# Temperature Data : 2013

# In[ ]:


import pandas as pd


# In[ ]:


temp = pd.read_csv('../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCountry.csv')


# In[ ]:


temp['dt']=pd.to_datetime(temp['dt'])
temp=temp.set_index('dt')


# In[ ]:


temp=temp.replace(to_replace='Bosnia And Herzegovina',value='Bosnia and Herzegovina')

covid_current=covid_pd.groupby('Country').agg({'Confirmed':'max','Deaths':'max','Recovered':'max'}).reset_index()

covid_current['Remain']=covid_current['Confirmed']-covid_current['Recovered']-covid_current['Deaths']
covid_current['%Recovered Rate']=covid_current['Recovered']/covid_current['Confirmed']*100
covid_current['%Deaths']=covid_current['Deaths']/covid_current['Confirmed']*100
covid_current.replace(np.inf,0,inplace=True)

top_n=100
top_country=covid_current.sort_values(by=['Confirmed'],ascending=False)['Country'].values[:top_n]


# In[ ]:


country_lat_long=lat_long.groupby('Country/Region').agg({'Lat':'mean','Long':'mean'}).reset_index()


# In[ ]:


covid_map=covid_current.merge(country_lat_long,left_on='Country',right_on='Country/Region',how='left')
#Change country name to match with wolrd json file
covid_map=covid_map.replace(to_replace='United States',value='United States of America')
covid_map=covid_map.sort_values(by=['%Recovered Rate'],ascending=False)


# In[ ]:


covid_map.head(10)


# In[ ]:


#world_geo="/kaggle/input/geo-json-world/world-countries.json"
world_geo="/kaggle/input/world-countries/world-countries.json"


# In[ ]:


covid_map.head(5)


# In[ ]:


map = folium.Map(location=[10, 15], 
                 tiles = "CartoDB dark_matter",
                 detect_retina = True,
                 zoom_start=2)
#Remove Top eg (China , Small country)
folium.Choropleth(
    geo_data=world_geo,
    name='choropleth',
    key_on='feature.properties.name',
    data=covid_map,
    columns=['Country','%Recovered Rate'],
    fill_color='PuBu'
).add_to(map)


map


# Map show %recovered Rate 

# In[ ]:


#Screen outlier : few sampling of confime)
covid_death=covid_map.sort_values(by=['%Deaths'],ascending=False)
covid_death=covid_death[covid_death['Confirmed']>50]


# In[ ]:


covid_death[covid_death['Confirmed']>50].sort_values(by=['%Deaths'],ascending=False).head()


# In[ ]:


map = folium.Map(location=[10, 15], 
                 tiles = "CartoDB dark_matter",
                 detect_retina = True,
                 zoom_start=2)

folium.Choropleth(
    geo_data=world_geo,
    name='choropleth',
    key_on='feature.properties.name',
    data=covid_death,
    columns=['Country','%Deaths'],
    fill_color='OrRd'
).add_to(map)

#fill_color: fill color code
#'BuGn', 'BuPu', 'GnBu', 'OrRd', 'PuBu', 'PuBuGn', 'PuRd', 
#'RdPu', 'YlGn', 'YlGnBu', 'YlOrBr', and 'YlOrRd'.


map


# %Death Rate

# Temperature on coutryies with COVID

# Temperature on Mar-20.
# Many counties have covid confirmed cases more than 1000 at beginig of Mar-20.

# In[ ]:


#Find average temperature on  back 20 years
a=np.array([])
covid_current['temperature']=0
for c in top_country:
    avg=np.mean((temp[(temp['Country']==c) & (temp.index.month.isin([3]))].                 groupby(pd.Grouper(freq='M')).agg({'AverageTemperature':'mean'}).                 reset_index()).dropna(subset=['AverageTemperature'])['AverageTemperature'].values[-20:])
    covid_current.loc[covid_current['Country']==c,['temperature']]=avg
    
    #if(math.isnan(avg)):
    #    print(c,avg)

ir_temp=np.mean((temp[(temp['Country']=='Ireland') & (temp.index.month.isin([3]))].             groupby(pd.Grouper(freq='M')).agg({'AverageTemperature':'mean'}).             reset_index()).dropna(subset=['AverageTemperature'])['AverageTemperature'].values[-20:])
covid_current.loc[covid_current['Country']=='Republic of Ireland',['temperature']]=ir_temp
covid_current.loc[covid_current['Country']=='North Ireland',['temperature']]=ir_temp

vatican_temp=np.mean((temp[(temp['Country']=='Italy') & (temp.index.month.isin([3]))].             groupby(pd.Grouper(freq='M')).agg({'AverageTemperature':'mean'}).             reset_index()).dropna(subset=['AverageTemperature'])['AverageTemperature'].values[-20:])

covid_current.loc[covid_current['Country']=='Vatican City',['temperature']]=vatican_temp

palestine_temp=np.mean((temp[(temp['Country']=='Israel') & (temp.index.month.isin([3]))].             groupby(pd.Grouper(freq='M')).agg({'AverageTemperature':'mean'}).             reset_index()).dropna(subset=['AverageTemperature'])['AverageTemperature'].values[-20:])

covid_current.loc[covid_current['Country']=='Palestine',['temperature']]=palestine_temp

denmark_temp=np.mean((temp[(temp['Country']=='Denmark (Europe)') & (temp.index.month.isin([3]))].             groupby(pd.Grouper(freq='M')).agg({'AverageTemperature':'mean'}).             reset_index().dropna(subset=['AverageTemperature'])['AverageTemperature'].values[-20:]))
#covid_current.loc[covid_current['Country']=='Denmark',['temperature']]=vatican_temp
covid_current.loc[covid_current['Country']=='Denmark',['temperature']]=denmark_temp

#Change country name to match with wolrd json file
covid_current=covid_current.replace(to_replace='United States',value='United States of America')


# If you run above cell and found error.
# 
# AttributeError: 'DataFrame' object has no attribute 'Grouper'
# 
# Please run import Cell again. It's will work.
# 

# In[ ]:


all_temp=temp[(temp.index>'2010-01-01')&(temp.index.month.isin([3]))].groupby('Country').agg({'AverageTemperature':'mean'}).reset_index().sort_values(by=['AverageTemperature'])
all_temp=all_temp.replace(to_replace='United States',value='United States of America')
all_temp_risk=all_temp[(all_temp['AverageTemperature']>-0) & (all_temp['AverageTemperature']<15)]


# In[ ]:


#Change country name to match with wolrd json file
all_temp=all_temp.replace(to_replace='United States',value='United States of America')


# In[ ]:


pd=covid_current.sort_values(by=['Confirmed'],ascending=False)
case=pd['Confirmed'].values
t=pd['temperature'].values
country=pd['Country'].values
plt.figure(figsize=(12, 8))

plt.scatter(t,case)
plt.xlabel('Celsius')
plt.ylabel('confirmed')
plt.yscale('log')
plt.axhline(y=100000,xmin= 0,color='g')
plt.axhline(y=1000,xmin= 0,color='g')

plt.axvline(x=0,ymin= 0,color='r')
plt.axvline(x=15,ymin= 0,color='r')
for i, txt in enumerate(pd['Country'][:15]):
    plt.annotate(txt, ( t[i],case[i]))
plt.show()


# Chart show relation between temperature.
# Temperature = Average Temperature in Febuary last 20 years.
# Temperature of High infected in many coutries are in range between 0 celsius to 15 celsius.

# In[ ]:


max_con=100000
min_con=10000
pd=covid_current[(covid_current['Confirmed']<100000) & (covid_current['Confirmed']>10000) & (covid_current['Country']!='China')]
case_low=pd['Confirmed'].values
temp_low=pd['temperature'].values
country_low=pd['Country'].values
total=pd.count()[0]
in_range=pd[(pd['temperature']>=-5) & (pd['temperature']<=10)].count()[0]
plt.figure(figsize=(10, 8))

plt.xlabel('Celsius')
plt.ylabel('confirmed')
plt.yscale('log')

plt.scatter(temp_low,np.log(case_low) )
for i, txt in enumerate(country_low):
    plt.annotate(txt, ( temp_low[i],np.log(case_low[i])))
plt.show()
print(in_range,"in",total,"countries of ",str(min_con),"-",str(max_con)," cases temperature range = -5 to 10 Celsius")


# In[ ]:


pd=covid_current[(covid_current['Confirmed']<10000) & (covid_current['temperature']<15) & (covid_current['temperature']>0) & (covid_current['Country']!='China')]
case=pd['Confirmed'].values
t=pd['temperature'].values
country=pd['Country'].values
plt.figure(figsize=(10, 8))
plt.xlabel('Celsius')
plt.ylabel('Confirmed)')
plt.scatter(t,case)
for i, txt in enumerate(country):
    plt.annotate(txt, ( t[i],case[i]))
plt.show()


# Above chart show country which less than 10000 cases and temperature ranging between 0-15 celsius.Mostly new countries need to monitor number of infected.

# In[ ]:


map = folium.Map(location=[10, 15], 
                 tiles = "CartoDB dark_matter",
                 detect_retina = True,
                 zoom_start=2)

folium.Choropleth(
    geo_data=world_geo,
    name='choropleth',
    key_on='feature.properties.name',
    data=covid_current,
    columns=['Country','temperature'],
    fill_color='PuRd'
).add_to(map)

#fill_color: fill color code
#'BuGn', 'BuPu', 'GnBu', 'OrRd', 'PuBu', 'PuBuGn', 'PuRd', 
#'RdPu', 'YlGn', 'YlGnBu', 'YlOrBr', and 'YlOrRd'.


map


# Temperature map by countrie with COVID.

# In[ ]:


plt.figure(figsize=(8, 12))
plt.barh(all_temp_risk['Country'],all_temp_risk['AverageTemperature'])
plt.show()


# In[ ]:


check_n=all_temp_risk.merge(covid_current,on='Country',how='left')
check_n[(check_n['Confirmed'].isnull()) & (check_n['AverageTemperature']>0) & (check_n['AverageTemperature']<10)]['Country']


# I try to find which countries no covid yet and temperature range in -10 to 10 in March, found mismatch typing and some countries are small islands. 
# 
# As I can see , risk countries will be Macedonia,Tajikistan,and Turkmenistan. (Coutry without covid on 13 Apr 20)
# 
# Anyway , Risk of covid spreading into these countries depend on outbound travelers controling in each country to protect virus from abroad. And Higher temperture in Summer will help to reduce severity of virus.

# In[ ]:


map = folium.Map(location=[10, 15], 
                 tiles = "CartoDB dark_matter",
                 detect_retina = True,
                 zoom_start=2)

folium.Choropleth(
    geo_data=world_geo,
    name='choropleth',
    key_on='feature.properties.name',
    data=all_temp,
    columns=['Country','AverageTemperature'],
    fill_color='PuRd'
).add_to(map)

#fill_color: fill color code
#'BuGn', 'BuPu', 'GnBu', 'OrRd', 'PuBu', 'PuBuGn', 'PuRd', 
#'RdPu', 'YlGn', 'YlGnBu', 'YlOrBr', and 'YlOrRd'.


map


# Temperature by Country: 
# 
# Risk Coutries with Temperature range 3 Celsius to 13 Celsius are in Europe,North America,Central Asia, and East Asia.

# Some coutries spread so fast , Some countries not. What is the factor?

# In[ ]:


country_covid_confirm=country_covid.pivot(index='ObservationDate',columns='Country',values='Confirmed').reset_index()
country_covid_confirm=country_covid_confirm.fillna(0)


# In[ ]:


top_n=10
top_country=covid_pd.groupby(['Country']).agg({'Remain':'max'}).reset_index().sort_values(by=['Remain'],ascending=False).head(top_n)


# In[ ]:


new_case=country_covid_confirm.iloc[:,1:].diff()
new_case.replace(np.nan,0,inplace=True)


# In[ ]:


start_top=1
for c in (top_country['Country'].values[start_top:]):
    plt.plot(new_case[c])
plt.gca().legend(top_country['Country'].values[start_top:top_n])
plt.xlabel('Days')
plt.ylabel('New Cases')
plt.show()


# In[ ]:


import pandas as pd
top_n=50
top_country=covid_pd.groupby(['Country']).agg({'Remain':'max'}).reset_index().sort_values(by=['Remain'],ascending=False).head(top_n)
a=np.array([])
for c in (top_country['Country'].values):
    day_data=covid_pd[covid_pd['Country']==c].groupby('ObservationDate').agg({'Confirmed':'max'}).sort_values(by=['ObservationDate']).reset_index()
    day_accum=day_data['Confirmed'].values
    min_date=day_data['ObservationDate'][0]
    max_change=0
    if day_accum[0]>0:
        d=day_data['ObservationDate'][0]
    else:
        d=''
    for i in range(1,len(day_accum)):
        if day_accum[i-1]>0 and (day_accum[i]-day_accum[i-1])/day_accum[i-1] > max_change:
            max_change=(day_accum[i]-day_accum[i-1])/day_accum[i-1]
            if d=='':
                d=day_data['ObservationDate'][i]
    a=np.append(a,[max_change,d])
a=a.reshape((-1,2))
country=top_country['Country'].values.reshape((-1,1))
s=np.concatenate([country,a],axis=1)
spread=pd.DataFrame(s,columns=['Country','max_speed','Start_date'])
spread['max_speed']=pd.to_numeric(spread['max_speed'])
spread=spread.sort_values(by=['max_speed'],ascending=False)


# In[ ]:


map = folium.Map(location=[10, 15], 
                 tiles = "CartoDB dark_matter",
                 detect_retina = True,
                 zoom_start=2)

folium.Choropleth(
    geo_data=world_geo,
    name='choropleth',
    key_on='feature.properties.name',
    data=spread[2:],
    columns=['Country','max_speed'],
    fill_color='PuBu'
).add_to(map)


map


# Find max increase compare yesterday by contries
# 
# Max((Confirmed Case today)-(Confirmed Case yesterday))/(Confirmed Case yesterday)
# 
# *Can't distinguish color scale. So I excluse Bahrain & Kuwait which is 1st and 2nd max speed.

# In[ ]:


covid_current=pd.merge(covid_current, spread, on='Country',how='left')


# Population Density

# In[ ]:


pop_dense=pd.read_csv('/kaggle/input/migration-data-worldbank-1960-2018/migration_population.csv')


# In[ ]:


pop_data=pop_dense[pop_dense['year']==2018].groupby(['country','year']).agg({'population':'max','pop_density':'max','region':'max',
                                           'incomeLevel':'max','lendingType':'max',
                                            'longitude':'mean','latitude':'mean'}).reset_index()


# In[ ]:


pop_data.head()


# In[ ]:


lat_long.head()


# In[ ]:


covid_current=pd.merge(covid_current, lat_long, left_on='Country',right_on='Country/Region')


# In[ ]:


covid_current=covid_current.iloc[:,:14]


# In[ ]:


covid_current=pd.merge(covid_current, pop_data, left_on='Country',right_on='country',how='left')
covid_map=covid_map.dropna()


# In[ ]:


map = folium.Map(location=[10, 15], 
                 tiles = "CartoDB dark_matter",
                 detect_retina = True,
                 zoom_start=2)

folium.Choropleth(
    geo_data=world_geo,
    name='choropleth',
    key_on='feature.properties.name',
    data=all_temp,
    columns=['Country','AverageTemperature'],
    fill_color='PuRd'
).add_to(map)

#fill_color: fill color code
#'BuGn', 'BuPu', 'GnBu', 'OrRd', 'PuBu', 'PuBuGn', 'PuRd', 
#'RdPu', 'YlGn', 'YlGnBu', 'YlOrBr', and 'YlOrRd'.

for i in range(len(covid_map)):
      folium.CircleMarker(location = [covid_map['Lat'].iloc[i],covid_map['Long'].iloc[i]], 
                        radius = np.log(covid_map['Confirmed'].iloc[i])*2, 
                        color = '#E80018', 
                        fill_opacity = 0.7,
                        weight = 2, 
                        fill = True
                       ,fillColor = '#E80018'
                         ).add_to(map)

map


# Temperature vs Confirmed cases

# In[ ]:


plt.figure(figsize=(16, 12))
plt.scatter(np.log(covid_current['Confirmed']),np.log(covid_current['pop_density']))
for i, txt in enumerate(covid_current['Country']):
    plt.annotate(txt, ( np.log(covid_current['Confirmed'][i]),np.log(covid_current['pop_density'][i])))
plt.xlabel('Confirmed')
plt.ylabel('pop_density')
plt.show()


# Calculate Correlation : Method = pearson , kendall , and spearman

# In[ ]:


covid_current.corr(method ='pearson').iloc[:,6:]


# In[ ]:


covid_current.corr(method ='spearman').iloc[:,6:]


# In[ ]:


covid_current.corr(method ='kendall').iloc[:,6:]


# In[ ]:


plt.figure(figsize=(16, 12))
plt.scatter(np.log(covid_current['population']),covid_current['%Recovered Rate'])
for i, txt in enumerate(covid_current['Country']):
    plt.annotate(txt, ( np.log(covid_current['population'][i]),covid_current['%Recovered Rate'][i]))


# In[ ]:


covid_current.head()


# In[ ]:


covid_current[covid_current['Country']!='China'].groupby('region').agg({'Confirmed':'sum','Recovered':'sum','Deaths':'sum'})


# In[ ]:


rate=covid_current[covid_current['Country']!='China'].groupby('region').agg({'%Recovered Rate':'mean'}).reset_index().sort_values(by=['%Recovered Rate'])


# In[ ]:


plt.figure(figsize=(12, 8))
plt.barh(rate['region'],rate['%Recovered Rate'])
plt.xlabel('%Recovered')
plt.show()


# In[ ]:


rate=covid_current[covid_current['Country']!='China'].groupby('region').agg({'%Deaths':'mean'}).reset_index().sort_values(by=['%Deaths'])


# In[ ]:


plt.figure(figsize=(12, 8))
plt.barh(rate['region'],rate['%Deaths'])
plt.xlabel('%Deaths')
plt.show()


# In[ ]:


covid_current.groupby('incomeLevel').agg({'Confirmed':'sum','Recovered':'sum','Deaths':'sum'})


# Mostly of upper middle income come from China. High income also impact a lot.

# In[ ]:


rate_income=covid_current.groupby('incomeLevel').agg({'%Recovered Rate':'mean','%Deaths':'mean'}).reset_index().sort_values(by=['%Recovered Rate'])


# In[ ]:


plt.figure(figsize=(12, 8))
plt.barh(rate_income['incomeLevel'],rate_income['%Recovered Rate'])

plt.xlabel('%Recovered')

plt.show()


# In[ ]:


rate_income=covid_current.groupby('incomeLevel').agg({'%Recovered Rate':'mean','%Deaths':'mean'}).reset_index().sort_values(by=['%Deaths'])


# In[ ]:


plt.figure(figsize=(12, 8))
plt.barh(rate_income['incomeLevel'],rate_income['%Deaths'])
plt.xlabel('%Deaths')
plt.show()


# #Urbanize
# https://www.cia.gov/library/publications/the-world-factbook/fields/349.html

# In[ ]:


urban_pd=pd.read_csv('/kaggle/input/urbanize-percentage-by-country-2020/urban percentage by country.csv')


# In[ ]:


urban_pd.rename(columns={'COUNTRY':'Country'},inplace=True)


# In[ ]:


urban_pd.head()


# In[ ]:


covid_current=pd.merge(covid_current, urban_pd, on='Country',how='left')


# In[ ]:


covid_current.head()


# Calculate Correlation

# In[ ]:


covid_current.corr(method ='pearson').iloc[:,6:]


# In[ ]:


covid_current.corr(method ='kendall').iloc[:,6:]


# In[ ]:


covid_current.corr(method ='spearman').iloc[:,6:]


# In[ ]:


plt.figure(figsize=(16, 12))
plt.scatter(np.log(covid_current['Confirmed']),covid_current['%Urban'])
for i, txt in enumerate(covid_current['Country']):
    plt.annotate(txt, ( np.log(covid_current['Confirmed'][i]),covid_current['%Urban'][i]))
plt.xlabel('log(Confimred)')
plt.ylabel('%Urban')
plt.show()


# In[ ]:


plt.figure(figsize=(16, 12))
plt.scatter(covid_current['%Recovered Rate'],covid_current['%Urban'])
for i, txt in enumerate(covid_current['Country']):
    plt.annotate(txt, ( covid_current['%Recovered Rate'][i],covid_current['%Urban'][i]))
plt.xlabel('%Recovered Rate')
plt.ylabel('%Urban')
plt.show()


# In[ ]:




