# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib import animation

pcm = pd.read_csv('../input/300k.csv', low_memory=False)


time_groups = pcm.groupby('appearedHour')
plt.figure(1, figsize=(30,20))
m = map = Basemap(
    projection='merc', 
    llcrnrlat=-60,
             urcrnrlat=65,
             llcrnrlon=-180,
             urcrnrlon=180,
             lat_ts=0,
    resolution='i')
m.fillcontinents(color='#191919',lake_color='#000000') # dark grey land, black lakes
m.drawmapboundary(fill_color='#000000')                # black background
m.drawcountries(linewidth=0.1, color="w")              # thin white line for country borders

x,y = m(0, 0)
point = m.plot(x, y, 'o', markersize=2, color='b')[0]
def init():
    point.set_data([], [])
    return point,

def animate(i):
    lon = time_groups.get_group(i)['longitude'].values
    lat = time_groups.get_group(i)['latitude'].values
    x, y = m(lon ,lat)
    point.set_data(x,y)
    plt.title('Pocemon activity at %2d:00' % (i))
    return point,

output = animation.FuncAnimation(plt.gcf(), animate, init_func=init, frames=24, interval=500, blit=True, repeat=True)
output.save("pocemon.gif", writer='imagemagick')
plt.show()
