#!/usr/bin/env python
# coding: utf-8

# # Airpoclypse
# Air pollution is a critical problem all across the globe. We, humans, have ignored its consequences for so long. The natural disasters caused as a result dramatically changing climate, the rising temperature, untimely and fierce rains, all are consequences of our own ignorance. Trying to reduce world pollution is no easy task, and it can only be achieved by the co-operation of every single person in the world. 
# 
# ![](https://i.gifer.com/3zdg.gif)

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import random

city_hour = pd.read_csv('../input/air-quality-data-in-india/city_hour.csv')
station_hour = pd.read_csv('../input/air-quality-data-in-india/station_hour.csv')
station_day = pd.read_csv('../input/air-quality-data-in-india/station_day.csv')
city_day = pd.read_csv('../input/air-quality-data-in-india/city_day.csv')
stations = pd.read_csv('../input/air-quality-data-in-india/stations.csv')
city_day['Date'] = pd.to_datetime(city_day['Date'],dayfirst = True)
city_hour['Datetime'] = pd.to_datetime(city_hour['Datetime'],dayfirst = True)

city_day['year'] = [d.year for d in city_day.Date]
city_day['month'] = [d.strftime('%b') for d in city_day.Date]
city_hour['hours'] = [d.hour for d in city_hour.Datetime]
city_day.fillna(method='bfill',inplace=True)
city_hour.fillna(method='bfill',inplace=True)


# ## Trends of Pollutants

# In[ ]:


def trend_plot(value):   
    years = city_day['year'].unique()
    fig, axes = plt.subplots(1, 3, figsize=(14,6), dpi= 80)
    sns.boxplot(x='year', y=value, data=city_day, ax=axes[0])
    sns.pointplot(x='month', y=value, data=city_day.loc[~city_day.year.isin([1991, 2008]), :],ax=axes[1])
    sns.pointplot(x='hours', y=value, data=city_hour)
    axes[0].set_title('Yearly Trend', fontsize=18); 
    axes[1].set_title('Monthy Trend', fontsize=18)
    axes[2].set_title('Hourly Trend',fontsize =18)
    plt.show()
values = ['NO','NO2','NH3','CO','SO2','O3']
for value in values:
    trend_plot(value)


# In[ ]:


DAYS = ['Sun.', 'Mon.', 'Tues.', 'Wed.', 'Thurs.', 'Fri.', 'Sat.']
MONTHS = ['Jan.', 'Feb.', 'Mar.', 'Apr.', 'May', 'June', 'July', 'Aug.', 'Sept.', 'Oct.', 'Nov.', 'Dec.']
colors = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

def date_heatmap(series, start=None, end=None, mean=False, ax=None, **kwargs):
    dates = series.index.floor('D')
    group = series.groupby(dates)
    series = group.mean() if mean else group.sum()
    start = pd.to_datetime(start or series.index.min())
    end = pd.to_datetime(end or series.index.max())
    end += np.timedelta64(1, 'D')
    start_sun = start - np.timedelta64((start.dayofweek + 1) % 7, 'D')
    end_sun = end + np.timedelta64(7 - end.dayofweek - 1, 'D')
    num_weeks = (end_sun - start_sun).days // 7
    heatmap = np.zeros((7, num_weeks))
    ticks = {}  # week number -> month name
    for week in range(num_weeks):
        for day in range(7):
            date = start_sun + np.timedelta64(7 * week + day, 'D')
            if date.day == 1:
                ticks[week] = MONTHS[date.month - 1]
            if date.dayofyear == 1:
                ticks[week] += f'\n{date.year}'
            if start <= date < end:
                heatmap[day, week] = series.get(date, 0)
    y = np.arange(8) - 0.5
    x = np.arange(num_weeks + 1) - 0.5
    ax = ax or plt.gca()
    mesh = ax.pcolormesh(x, y, heatmap, **kwargs)
    ax.invert_yaxis()
    ax.set_xticks(list(ticks.keys()))
    ax.set_xticklabels(list(ticks.values()))
    ax.set_yticks(np.arange(7))
    ax.set_yticklabels(DAYS)
    plt.sca(ax)
    plt.sci(mesh)
    return ax
def calender_map(data,value,year,color):
    figsize = plt.figaspect(7 / 56)
    fig = plt.figure(figsize=figsize)
    ax = date_heatmap(data, edgecolor='black')
    tick=[]
    for i in range(int(city_day[value].min()),int(city_day[value].max()),100):
        tick.append(i)
    plt.colorbar(ticks=tick, pad=0.02)
    cmap = mpl.cm.get_cmap(color, 100)
    plt.set_cmap(cmap)
    plt.clim(city_day[value].min(), city_day[value].max())
    ax.set_aspect('equal')
    plt.title('{} calender map in the year {}'.format(value,year),fontsize=15)
    plt.show()

def calender_plot(value,year,color):
    df = city_day[['Date',value,'year']]
    df=df[df['year'] == year]
    df.set_index('Date',inplace=True)
    date = list(df.index)
    data = list(df[value])
    data = pd.Series(data)
    data.index = date
    calender_map(data,value,year,color)
years = list(city_day['year'].unique())


# ## Calender Map of AQI and PM2.5

# In[ ]:


for year in years:
    calender_plot('AQI',year,'PuBuGn')
for year in years:
    calender_plot('PM2.5',year,'BuPu')


# ## Coorelation between the Pollutants

# In[ ]:


variables = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','Xylene']

fig = plt.figure(figsize=(16,34))
for variable,num in zip(variables, range(1,len(variables)+1)):
    ax = fig.add_subplot(6,2,num)
    sns.scatterplot(variable, 'AQI', hue='year', data=city_day)
    plt.title('Relation between {} and AQI'.format(variable))
    plt.xlabel(variable)
    plt.ylabel('AQI')


# In[ ]:


plt.figure(figsize=(20, 20));
sns.heatmap(city_day.corr(),annot=True);


# # Delhi
# ## The Impact of Lockdown on Air Quality 

# ## City-wise Pollitants Insights

# In[ ]:


fig, axes= plt.subplots(figsize=(20, 12), ncols=5)
city_wise_max_so2 = city_day[['City','SO2']].dropna().groupby('City').mean().sort_values(by='SO2')
city_wise_max_no2 = city_day[['City','NO2']].dropna().groupby('City').mean().sort_values(by='NO2')
city_wise_max_pm25 = city_day[['City','PM2.5']].dropna().groupby('City').mean().sort_values(by='PM2.5')
city_wise_max_aqi = city_day[['City','AQI']].dropna().groupby('City').mean().sort_values(by='AQI')
city_wise_max_pm10 = city_day[['City','PM10']].dropna().groupby('City').mean().sort_values(by='PM10')

sns.barplot(x='SO2', y=city_wise_max_so2.index, data=city_wise_max_so2, ax=axes[0])
axes[0].set_title("Average SO2 Observed in a City")

sns.barplot(x='NO2', y=city_wise_max_no2.index, data=city_wise_max_no2, ax=axes[1])
axes[1].set_title("Average NO2 observed in a City")

sns.barplot(x='PM2.5', y=city_wise_max_pm25.index, data=city_wise_max_pm25, ax=axes[2])
axes[2].set_title("Average PM2.5 observed in a City")

sns.barplot(x='AQI', y=city_wise_max_aqi.index, data=city_wise_max_aqi, ax=axes[3])
axes[3].set_title("Average AQI observed in a city")

sns.barplot(x='PM10', y=city_wise_max_pm10.index, data=city_wise_max_pm10, ax=axes[4])
axes[4].set_title("Average pm2_5 observed in a city")
plt.tight_layout()


# In[ ]:


def cityandYear(indicator1,indicator2):
    fig, axes= plt.subplots(figsize=(20, 12), ncols=2);
    plt.figure(figsize=(20, 20));
    hmap = sns.heatmap(
        data=city_day.pivot_table(values=indicator1, index='City', columns='year', aggfunc='mean', margins=True),
               annot=True, linewidths=.5, cbar=True, square=True, cmap='inferno', cbar_kws={'label': "Annual Average"},ax = axes[0]);
    hmap = sns.heatmap(
        data=city_day.pivot_table(values=indicator2, index='City', columns='year', aggfunc='mean', margins=True),
               annot=True, linewidths=.5, cbar=True, square=True, cmap='inferno', cbar_kws={'label': "Annual Average"},ax = axes[1]);
    
    axes[0].set_title("{} by City and Year".format(indicator1),fontsize=15);
    axes[1].set_title("{} by City and Year".format(indicator2),fontsize=15);


# In[ ]:


cityandYear('NO2','SO2')
cityandYear('PM2.5','PM10')
cityandYear('NH3','AQI')


# In[ ]:


def lockdownEffect(city,value):
    df = city_day[city_day['Date'] > '3-1-2020']
    df = df[df['City'] == city]
    fig, ax1 = plt.subplots(figsize= (15,5));
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    ax1.set_ylabel(value)
    ax1.bar(df['Date'],df[value]);
    df = city_day[city_day['Date'] > '3-24-2020']
    df = df[df['City'] == city]
    ax1.bar(df['Date'],df[value],color='red');
    plt.title('{} {}'.format(value, city))
def lockdown_hourly_effect(city,value1,value2):
    city_df = city_hour[city_hour['City'] == city]
    fig, axes = plt.subplots(1, 2, figsize=(14,6), dpi= 80)
    
    sns.pointplot(x='hours', y=value1, data=city_df,label = 'After Lockdown',color='blue',ax=axes[0])
    sns.pointplot(x='hours', y=value2, data=city_df,label = 'After Lockdown',color='blue',ax=axes[1])
    axes[0].set_title(city +' '+str(value1)+ ' Hourly Trend', fontsize=18); 
    axes[1].set_title(city +' '+str(value2)+ ' Hourly Trend', fontsize=18); 
    


# # Delhi
# ![](https://cdn.britannica.com/37/189837-050-F0AF383E/New-Delhi-India-War-Memorial-arch-Sir.jpg)
# 
# ## The Impact of Lockdown on Air Quality

# In[ ]:


lockdownEffect('Delhi','PM2.5')
lockdownEffect('Delhi','PM10')
lockdownEffect('Delhi','NO2')
lockdownEffect('Delhi','SO2')


# During the lockdown period, as a result of combination of reduced vehicles on the road, functioning of only
# essential commercial units and prevailing weather conditions, significant reduction in PM2.5, PM10 and NO2
# levels are observed.

# In[ ]:


lockdown_hourly_effect('Delhi','PM2.5','PM10')
lockdown_hourly_effect('Delhi','SO2','NO2')


# # Mumbai
# 
# ![](https://cdn.britannica.com/26/84526-050-45452C37/Gateway-monument-India-entrance-Mumbai-Harbour-coast.jpg)
# 
# ## The Impact of Lockdown on Air Quality
# 
# 

# In[ ]:


lockdownEffect('Mumbai','PM2.5')
lockdownEffect('Mumbai','PM10')
lockdownEffect('Mumbai','NO2')
lockdownEffect('Mumbai','SO2')


# There is a significant decrease in PM2.5, PM10, NO2 levels. Out of trend there is a increase in SO2 level.

# In[ ]:


lockdown_hourly_effect('Mumbai','PM2.5','PM10')
lockdown_hourly_effect('Mumbai','SO2','NO2')


# # Bangalore
# 
# ![](https://cms.qz.com/wp-content/uploads/2017/08/bangalore1-reuters-traffic-moves-along-a-road-in-the-southern-indian-city-of-bangalore-december-14-2005.jpg?quality=75&strip=all&w=1600&h=900&crop=1)
# 
# ## The Impact of Lockdown on Air Quality

# In[ ]:


lockdownEffect('Bengaluru','PM2.5')
lockdownEffect('Bengaluru','PM10')
lockdownEffect('Bengaluru','NO2')
lockdownEffect('Bengaluru','SO2')


# In[ ]:


lockdown_hourly_effect('Bengaluru','PM2.5','PM10')
lockdown_hourly_effect('Bengaluru','SO2','NO2')


# # Chennai
# 
# ![](https://img.dtnext.in/Images/Article/201611130048195611_Chennai-Central-gets-firstever-station-director_SECVPF.gif)
# 
# ## The Impact of Lockdown on Air Quality

# In[ ]:


lockdownEffect('Chennai','PM2.5')
lockdownEffect('Chennai','PM10')
lockdownEffect('Chennai','NO2')
lockdownEffect('Chennai','SO2')


# In[ ]:


lockdown_hourly_effect('Chennai','PM2.5','PM10')
lockdown_hourly_effect('Chennai','SO2','NO2')

