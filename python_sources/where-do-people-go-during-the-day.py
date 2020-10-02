#!/usr/bin/env python
# coding: utf-8

# My goal is to find a way to explore **spatio-temporal patterns** in the NYC Yellow Taxi trips such as: Where do people go and when? Where should I cab driver head at each moment to maximize their pay per hour/mile? What commute routes could benefit from extra public transportation and what hours? Here, you can find an example analyzing where people go during a day time lapse.
# 
# **Spatial analysis:** We use KNN regression to retrieve the scatter geolocations of the cab rides into a map. Additionally, a mask with the NYC boundary can be used. Such mask can be easily obtained using fiona package, shapely and OpenData GIS data. However, this is left out of this script due to the impossibility of adding external data:
# 
# https://data.cityofnewyork.us/City-Government/Congressional-Districts/qd3c-zuu7/data
# 
# **Temporal analysis:** We use a 'Gated' approach: grouping the data points in time frames by time of the day and apply temporal smoothing to eliminate further noisy reconstructions. Notice this approach could be used for different time analysis, for example, weekly patterns.
# 
# **Tools:** For this analysis, Pandas was used to handle the data and easily group it in time frames. Also, sklearn.neighbors' of KNN was used to perform the spatial regression. Finally, the result is saved as a .gif short movie using matplotlib.animation.

# **1- Import libraries:**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from sklearn import neighbors
from PIL import Image,ImageDraw,ImageFont


# **2- Define KNN parameters, load data and define the spatial grid:**

# In[ ]:


## KNN PARAMETERS:
n_neighbors =100 # this parameter is optimized for 512x512 resolution and 30m temporal windows
weights = 'distance'

## LOAD DATA:
fields = ['pickup_datetime','pickup_latitude','pickup_longitude','dropoff_datetime','dropoff_latitude','dropoff_longitude']
parse_dates = ['pickup_datetime','dropoff_datetime']
df = pd.read_csv('../input/3march.csv',usecols=fields,parse_dates=parse_dates)
df.dropna(how='any',inplace=True)
dfsave = df

# DEFINE THE SPATIAL GRID
ymax = 40.85
ymin = 40.65
xmin = -74.06
xmax = xmin +(ymax-ymin)
X,Y = np.mgrid[xmin:xmax:512j,ymin:ymax:512j] # the spatial resolution can be tunned up 
positions = np.vstack([X.ravel(),Y.ravel()])


# **3- Estimate the local passenger flux:**
# 
# The flux of passengers in an area is determined by doing KNN over the sample points. Each ride corresponds to two sample points: pickup and drop off. The value of each sample point corresponds to +1 for pickup points and -1 for drop offs. Essentially, KNN performs a weighted average of the cabs arriving and leaving the area, which is an estimation of the passenger flux.
# 
# The data is grouped by time of the day in 30 minute periods. For each time slot, a KNN reconstruction is obtained. In this case, only data from weekdays in march is used. It is likely that passenger flux will change in the weekends.
# 
# Due to limitations on the execution time, we are running only the frames between 4AM and 10AM, which successfully capture the morning rush hour.

# In[ ]:


# RECONSTRUCT THE MAP FOR EVERY TIME FRAME 
Zs = []
time_step = 30 #time gates of 30 minutes
for h in range(4,10): #only between 4AM anbd 10AM
        for m in np.arange(0,60,time_step).astype(int):
                df = dfsave[dfsave.pickup_datetime.dt.weekday<5] # select only weekdays
                df = dfsave.groupby(dfsave.pickup_datetime.dt.hour).get_group(h)
                df = df[(df.pickup_datetime.dt.minute>=m) & (df.pickup_datetime.dt.minute<(m+time_step))]
                values_pickup = np.vstack([df.pickup_longitude.values,df.pickup_latitude.values])
                values_dropoff = np.vstack([df.dropoff_longitude.values,df.dropoff_latitude.values])
                values = np.hstack([values_pickup,values_dropoff])
                targets = np.hstack([np.ones((1,values_pickup.shape[1])),-np.ones((1,values_dropoff.shape[1]))])
                knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
                Z = np.reshape(knn.fit(values.T,targets.T).predict(positions.T),X.shape)
                Zs.append(Z)


# **4- Reconstruct a movie with the time frames:**
# 
# Additionally, we introduce a temporal smoothing between consecutive frames under the assumption that they behave similarly. This further reduces the noise, which is important when you reduce the time intervals, since each frame has less points.

# In[ ]:


Writer = animation.writers['imagemagick']
writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
fig = plt.figure()
movie = []
h = 0 
m = -30 
for i in range(len(Zs)):
        m = m + 30
        if m==60:
                h = h+1 
                m = 0 
        Z = 0.25*Zs[i-1]+0.5*Zs[i]   # temporal smoothing
        if (i+1)==len(Zs):
                Z = Z+0.25*Zs[0]
        else:
                Z = Z+0.25*Zs[i+1]
        Z = np.rot90(Z)
        frame = plt.imshow(Z,extent=[xmin,xmax,ymin,ymax],clim=[-1,1],cmap='RdBu',animated=True)
        movie.append([frame])

ani = animation.ArtistAnimation(fig,movie, interval=100, blit=False,repeat_delay=0)
ani.save('./animation_out.gif', writer=writer)


# **5- Reconstructed image using NYC boundary GIS info:**
# 
# 
# Finally, you can obtain a more visually appealing result when you use a mask with the NYC boundary outline (to ignore pints on the water) and a higher resolution 512x512 pixels. Here is the result in that case, for the 24h reconstructed:
# ![Reconstructed movie using NYC boundary info][1]
#   [1]: https://s2.postimg.org/u1omxi2g7/animation.gif?dl=1&_ga=1.161350261.950108241.1489858682
# 
# .
# 
# 
# In conclusion, this representation can be used to visualize spatio-temporal patterns such as the passenger flux across the day. In this example, it is clear that people head to some Manhattan areas during the morning rush hour from Queens and Brooklyn, especially to Midtown. In the evening, the process is inverse  but way more disperse in time.
# 
# It is worth noticing that this approach can be used to map any other metric: tip percentage, trip distance, number of passengers, etc. by just changing the target value used in the KNN. Also, this same approach can be used with different time scope: weekly/monthly patterns.
