#!/usr/bin/env python
# coding: utf-8

# # Time Collision Correlation
# 
# I am new to data science and want to take a look at the date/time collisions happen. Any advise welcome. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


vehicle_data = pd.read_csv('../input/database.csv')
vehicle_data.dtypes


# # Plot collision rat over the months

# In[ ]:


vehicle_data['DATE'] = pd.to_datetime(vehicle_data['DATE'])
vehicle_data['DAY_OF_WEEK'] = vehicle_data['DATE'].dt.dayofweek


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

date_count = vehicle_data.groupby('DATE').DATE.count()

values = date_count.values
dates = date_count.index

mean = sum(values)/len(date_count)
variance = np.sqrt(sum((values-mean)**2)/len(date_count))

plt.bar(dates, values)
plt.title('Amount of Collisions')
plt.xticks(rotation='vertical')
plt.show()


# Looking for ways to normalize the data I found two on [wikipedia](https://en.wikipedia.org/wiki/Normalization_(statistics)).
# 
# # Standard Score

# In[ ]:


standard_date_count = (values-mean)/variance
plt.bar(dates, standard_date_count)
plt.title('Normalized Amount of Collisions with Standard Score')
plt.xticks(rotation='vertical')
plt.show()


# # Feature Scaling

# In[ ]:


feature_date_count = (values-min(values))/(max(values)-min(values))
plt.bar(dates, feature_date_count)
plt.title('Normalized Amount of Collisions with Feature Scaling')
plt.xticks(rotation='vertical')
plt.show()


# Looking at this bar chart we can see that the data is through all 4 months equaly distributed. We have 3 spikes. Need to figure out what happend back then.
# 
# More interisting in my point of view are the groups of four to five regularly piking above two other. Are these the weekends?
# 
# # Day of week correlated to crashes?

# In[ ]:


day_of_week_count = vehicle_data.groupby('DAY_OF_WEEK').DAY_OF_WEEK.count()
day_of_week = day_of_week_count.index
count = day_of_week_count.values

plt.bar(day_of_week, count)
plt.xticks(day_of_week+0.4, ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.show()


# It looks like we are driving on Fridays the most aggressive, but Saturdays and Sundays are chilled. 
# 
# What about time? What impact does the night do? What about 3-7am. Maybe drunken people are  more into collisions.
# 
# # Impact of Time

# In[ ]:


vehicle_data['TIME'] = pd.to_datetime(vehicle_data['TIME'])


# In[ ]:


time_count = vehicle_data.groupby('TIME').TIME.count()
time = time_count.index
count = time_count.values

plt.plot(time, count)
plt.xticks(rotation='vertical')


# First of all we see that the NYPD do not take the time accuratly. To overcome this problem, we have to convert the time into quarters.

# In[ ]:


import datetime 

def map_to_quarter(time):
    minute = 0
    if time.minute < 7:
        minute = 0
    elif time.minute < 22:
        minute = 15
    elif time.minute < 37:
        minute = 30
    elif time.minute < 52:
        minute = 45
        
    return datetime.datetime(time.year, time.month, time.day, time.hour, minute)


def map_to_hour(time):
    minute = 0        
    return datetime.datetime(time.year, time.month, time.day, time.hour, minute)


# In[ ]:


vehicle_data['QUARTERS'] = vehicle_data['TIME'].apply(map_to_quarter)
vehicle_data['HOURS'] = vehicle_data['TIME'].apply(map_to_hour)


# In[ ]:


quarters_count = vehicle_data.groupby('QUARTERS').QUARTERS.count()
hour_count = vehicle_data.groupby('HOURS').HOURS.count()
quarters = quarters_count.index
count = quarters_count.values
plt.plot(quarters, count, label='Per Quarter')
plt.plot(hour_count.index, hour_count.values, label='Per Hour')
plt.xticks(rotation='vertical')
plt.legend(bbox_to_anchor=[1.4, 1])
plt.title('Amount of Collisions')
plt.show()


# As you can see we tend to collide with stuff on our way to work and back home. Looks like we are really under pressure in situations like that. But what if we only look at the weekends when we don't drive to work?

# In[ ]:



days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for day in range(7):
    quarters_count = vehicle_data.loc[vehicle_data.DAY_OF_WEEK == day].groupby('QUARTERS').QUARTERS.count()

    quarters = quarters_count.index
    count = quarters_count.values
    plt.plot(quarters, count, label=days[day])
    plt.xticks(rotation='vertical')

plt.legend()


# Yeah, we have it right their less collisions around 9 o'clock on the weekends, but more at night. 

# # Plot New York By Collisions
# 
# Lets have a look at the locations where we collide.I will plot the coordinates as heat maps to have a look where we crash most frequently.

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

lon = vehicle_data['LONGITUDE']
lon = lon.dropna()
lat = vehicle_data['LATITUDE']
lat = lat.dropna()

#plt.scatter(lon, lat)
#plt.ylim([40.4,41.0])
#plt.xlim([-74.4,-73.6])

heatmap, xedges, yedges = np.histogram2d(lon, lat, bins=50)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.clf()
plt.imshow(heatmap.T, extent=extent, origin='lower')
plt.title('Heat Map of NY Collisions')
plt.show()


# ![New York][1]
# 
# 
#   [1]: https://upload.wikimedia.org/wikipedia/commons/3/34/5_Boroughs_Labels_New_York_City_Map.svg

# 1. Manhattan
# 2. Brooklyn
# 3. Queens
# 4. Bronx
# 5. Staten Island

# Look like Manhattan is the place to be, if you want to crash your car into another one. But where does it really hurt?
# 
# ## Collisions with Injuries

# In[ ]:


# Headmap of Injured
people_injured = vehicle_data.loc[vehicle_data['PERSONS INJURED'] > 0]
lon = people_injured['LONGITUDE']
lon = lon.dropna()
lat = people_injured['LATITUDE']
lat = lat.dropna()

heatmap, xedges, yedges = np.histogram2d(lon, lat, bins=50)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.clf()
plt.imshow(heatmap.T, extent=extent, origin='lower')
plt.title('Heat Map of NY Collisions with Injured People')
plt.show()


# Manhattan still strong but Brooklyn getting close. Lets have a closer look at these two.

# In[ ]:


manhattan_and_brooklin = people_injured.loc[(people_injured['BOROUGH'] == 'MANHATTAN') | (people_injured['BOROUGH'] == 'BROOKLYN')]
lon = manhattan_and_brooklin['LONGITUDE']
lon = lon.dropna()
lat = manhattan_and_brooklin['LATITUDE']
lat = lat.dropna()

heatmap, xedges, yedges = np.histogram2d(lon, lat, bins=50)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.clf()
plt.imshow(heatmap.T, extent=extent, origin='lower')
plt.title('Manhattan and Brooklyn')
plt.show()


# Okay now we know that you should get near (-73.975| 40.76) But does it help? Nope! Who always knows his freaking coordinates? So better plot the dangerous Streets!

# # The 10 most dangerous streets of New York

# In[ ]:



streets = vehicle_data.groupby('ON STREET NAME')['ON STREET NAME']
count = streets.count()
amount_of_crashes = len(vehicle_data.index)
count.sort_values(ascending=False, inplace=True)
freq_streets = count.index[:10]
print(count[:10])


# And the winner is the Broadway, haunted by the Atlantic Avenue! 
# 
# Amusing that the Jamaica Avenue does not live a Bob Marley life. What ever. Lets have a look where the streets are.

# In[ ]:


i = 0
lons = []
lats = []
for hex in matplotlib.colors.cnames:
    street = freq_streets[i]
    lon = vehicle_data.loc[(vehicle_data['ON STREET NAME']==street) | (vehicle_data['CROSS STREET NAME']==street)].LONGITUDE.dropna().values
    lat = vehicle_data.loc[(vehicle_data['ON STREET NAME']==street) | (vehicle_data['CROSS STREET NAME']==street)].LATITUDE.dropna().values
    lons.extend(lon)
    lats.extend(lat)
    plt.scatter(lon, lat, color=hex, label=street)
    i += 1
    if i==len(freq_streets):
        break

plt.legend(bbox_to_anchor=[1.6, 1])
plt.show()


# In[ ]:



heatmap, xedges, yedges = np.histogram2d(lons, lats, bins=50)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.clf()
plt.imshow(heatmap.T, extent=extent, origin='lower')
plt.show()


# Manhattan is the most dangerous place in New York, especially the Broadway counting 766 collisions within 4 Months.
# 
# Next I would like to combine time and space! And do a heat map for every hour of the day. 

# In[ ]:


import matplotlib.figure as figure

def plot_day_hours(day=0):
    
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    fig = plt.figure(figsize=(10,12))
    title = 'Crashes over Time'
    if day > 0 :
        title = days[day]+ ' ' + title 
    fig.suptitle(title)
    
    #f, axes = fig.subplots(6, 4, sharex='col', sharey='row')
    times = vehicle_data.groupby('HOURS').HOURS.count().index


    for i, time in zip(list(range(24)), times):
        row = ((i / 7)+1) * 100
        column = i % 3 +1
        data = vehicle_data.loc[(vehicle_data.HOURS == time)]
        if day>0 :
            data = data.loc[(vehicle_data.DAY_OF_WEEK==day)]
            
        lon = data.LONGITUDE.dropna()
        lat = data.LATITUDE.dropna()
        heatmap, xedges, yedges = np.histogram2d(lon, lat, bins=50)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        #axes[row, column].clf()
        ax = fig.add_subplot(6,4,i+1)
        ax.set_title(str(i)+":00")
        ax.imshow(heatmap.T, extent=extent, origin='lower')
        ax.tick_params(axis='x', which='both', bottom='off', left='off', top='off', labelbottom='off')
        ax.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
   


# In[ ]:


plot_day_hours(-1)


# In[ ]:


plot_day_hours(5)
plot_day_hours(2)


# 
