#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import numpy.random
import matplotlib.pyplot as plt

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import random


# In[ ]:


import pandas as pd
df =  pd.read_csv("../input/new-york-bike-share/JC-201706-citibike-NewYork.csv")
df


# In[ ]:


file_dataNY=("../input/new-york-bike-share/JC-201706-citibike-NewYork.csv")
dataNY= [line.strip().split(',') for line in open(file_dataNY)]
remove_row0_NY=dataNY.pop(0)


# In[ ]:


start_ID_NY = []
end_ID_NY = []
start_station = []
end_station_NY = []

for i in range(len(dataNY)):
    start_ID_NY.append(int(dataNY[i][3]))
    end_ID_NY.append(int(dataNY[i][7]))
    start_station.append(str(dataNY[i][4]))
    end_station_NY.append(str(dataNY[i][8]))
print(min(start_ID_NY))
print(min(end_ID_NY))
print(max(start_ID_NY))
print(max(end_ID_NY))
start_ID_NY.count(3211) #check


# In[ ]:


def CountFrequency(my_list): 
    freq = {} 
    for items in my_list: 
        freq[items] = my_list.count(items) 
    t = []
    for key, value in freq.items():
        t.append((value, key))
        
    t.sort() #sorts list from smallest to largest
    return t

        
freqStartID = CountFrequency(start_station)
print(freqStartID)


# In[ ]:


top_ten = freqStartID[-10:]

freq_ofStation, station_name = zip(*top_ten)
print(freq_ofStation)
print(station_name)


# In[ ]:


plt.figure(figsize=(10,5))
plt.barh(station_name, freq_ofStation, align='center', color='#ADD8E6')
plt.xlabel('Frequency of Use')
plt.ylabel('Station Name')
plt.title('Frequency of Use of Start Station (June 2017)')

plt.show()


# In[ ]:


freqEnd_station = CountFrequency(end_station_NY)
top_tenEnd = freqEnd_station[-10:]

freq_endstation, Estation_name = zip(*top_tenEnd)
print(freq_endstation)
print(Estation_name)


# In[ ]:


plt.figure(figsize=(10,5))
plt.barh(Estation_name, freq_endstation, align='center', color='#ADD8E6')
plt.xlabel('Frequency of Use')
plt.ylabel('Station Name')
plt.title('Frequency of Use of End Stations (June 2017)')

plt.show()


# In[ ]:


df['starttime'] = pd.to_datetime(df.starttime) 
starttime= df.starttime.dt.hour
a= np.array(starttime)
print(a)
new = np.array(a).tolist()
new.count(0) #check


# In[ ]:


freq_start_hour = CountFrequency(new)


# In[ ]:


hourfrequency=("../input/hour-frequency/hour_freq.csv")
hour_frequency= [line.strip().split(',') for line in open(hourfrequency)]
print(hour_frequency)


# In[ ]:


hour = []
hour_freq = []
for i in range(len(hour_frequency)):
    hour.append(int(hour_frequency[i][0]))
    hour_freq.append(int(hour_frequency[i][1]))
    
plt.figure(figsize=(12,7))
plt.bar(hour, hour_freq, align='center', color='#FFE4B5')
plt.xlabel('Time of Day (hour)')
plt.ylabel('Frequency of Use')
plt.title('Frequency of Use of Station by Hour in the Day (June 2017)')

plt.show()


# In[ ]:


import pandas as pd
JC_201706_citibike-NewYork = pd.read_csv("../input/JC-201706-citibike-NewYork.csv")


# In[ ]:


import pandas as pd
hour_freq = pd.read_csv("../input/hour_freq.csv")

