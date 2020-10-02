#!/usr/bin/env python
# coding: utf-8

# # Animated map of US wildfires
# I wanted to test animated graphs and show how wildfires developed over the year. I got some good ideas from this [kernel](https://www.kaggle.com/pavelevap/global-warming-confirmed-basemap-animation). So in this kernel I want to create an animated map that shows how wildfires spread over the continent in 2015 with a frame for each day of the year.
# 
# The dataset contains the longitude and latitude for each wildfire as well as the day of the year when the fire was discovered and when it was contained. So there's no much feature engineering to do, other than make sure the values are valid.
# 
# Take a look at the final animation at the end of this notebook. Super interesting to see how the fires start on the east coast and then in the second half of the year how the west coast is troubled by wild fires.

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


# connect to sqlite database
import sqlite3
conn = sqlite3.connect("../input/FPA_FOD_20170508.sqlite", detect_types=sqlite3.PARSE_DECLTYPES)


# In[ ]:


# read all fires in the continental US (excluding Alaska) from 2015 into dataframe
df = pd.read_sql_query("SELECT LATITUDE, LONGITUDE, FIRE_SIZE_CLASS, DISCOVERY_DOY,     CONT_DOY from Fires WHERE FIRE_YEAR=2015 AND CONT_DOY IS NOT NULL     AND DISCOVERY_DOY <= CONT_DOY     AND DISCOVERY_DOY IS NOT NULL AND LATITUDE IS NOT NULL     AND LONGITUDE IS NOT NULL     AND LATITUDE BETWEEN 25 AND 50     AND LONGITUDE BETWEEN -125 AND -67;",conn)


# In[ ]:


# this function returns the location and size of wildfires active on day X of the year
def active_fires(doy):
    return df[(df['DISCOVERY_DOY'] <= doy) & (df['CONT_DOY'] >= doy)]


# In[ ]:


# import some libraries
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from mpl_toolkits.basemap import Basemap
from matplotlib import animation


# In[ ]:


# building the animation. This code will show a static image. The animation needs to be saved first and will then show correctly.
# The animation will show for each day all the wildfires that were active during that day. The markers are sized to distinguish 
# smaller fires from bigger ones. 
fig = plt.figure(figsize=(18, 16))
    
map = Basemap(projection='cyl',llcrnrlat=25,urcrnrlat=50,            llcrnrlon=-125,urcrnrlon=-67)
map.drawcoastlines(linewidth=0.25)
map.drawcountries(linewidth=0.25)
map.drawstates(linewidth=0.25)
map.drawmapboundary()
map.fillcontinents(color='lightgray', zorder=1)
scat = map.scatter([], [])
day_label = plt.text(-124,26,'Day 1',fontsize=15)
def anim(frame_number):
    day = frame_number+1
    fires_this_day = active_fires(day)
    xs, ys = map(fires_this_day['LONGITUDE'], fires_this_day['LATITUDE'])
    data = np.dstack((xs, ys))[0]
    scat.set_color('r')
    scat.set_alpha(0.5)
    scat.set_zorder(10)
    scat.set_sizes(fires_this_day['FIRE_SIZE_CLASS'].map({'A':1,'B':2,'C':4,'D':8,'E':16,'F':32,'G':64}))
    scat.set_offsets(data)
    day_label.set_text("Day {}".format(day))
    return scat,

ani = animation.FuncAnimation(fig, anim, interval=100, frames=365, repeat=False, blit=True)
plt.title('US Wildfires during the year 2015')
plt.show()


# In[ ]:


# save the animation. this can take some time
ani.save('animation.gif', writer='imagemagick', fps=3)


# In[ ]:


# and display the final animation
from IPython.display import HTML
import io
import base64

filename = 'animation.gif'

video = io.open(filename, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" style="border-top:=100px; border-bottom:-100px;"/>'''.format(encoded.decode('ascii')))


# In[ ]:




