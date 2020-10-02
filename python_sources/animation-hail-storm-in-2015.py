import datetime
import pandas as pd
import numpy as np
import re 
import matplotlib.pyplot as plt


from mpl_toolkits.basemap import Basemap, cm
from matplotlib import animation
import matplotlib as mpl


hail = pd.read_csv('../input/hail-2015.csv',  parse_dates=['X.ZTIME'])

# clean datetime data
hail['X.ZTIME'] = hail['X.ZTIME'].replace('[^0-9]', ' ', regex=True)
hail['X.ZTIME'] = hail['X.ZTIME'].replace(' ', '', regex=True)

# we need to get only month and day
hail['month'] = hail['X.ZTIME'].map(lambda x: x[4:6])
hail['day'] = hail['X.ZTIME'].map(lambda x: x[6:8])

# useful if we want to set period for each day
hail['day_month'] = hail.day.apply(lambda x: str(x)) + hail.month.apply(lambda x: str(x)) 

# draw animation for each month
def draw_per_month():
    # period this iterator, it is used in the function animate(i) 
    period = hail.month.unique()

    # draw plot
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)

    # draw map of USA
    m = Basemap(llcrnrlon=-130,llcrnrlat=13,urcrnrlon=-60,urcrnrlat=52,projection='merc')
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    m.shadedrelief()

    # set data 
    x,y = m(0, 0)
    draw = m.plot(x, y,  'o',  markersize=1, color='white')[0]

    # make our animation
    def animate(i):
        p = hail[hail.month == i]

        fig.suptitle('Hail in {} month, 2015 year'.format(i), fontsize=22)
        lon = p.LON.values
        lat = p.LAT.values
        x, y = m(lon ,lat)
        draw.set_data(x,y)
        return draw,

    output = animation.FuncAnimation(plt.gcf(), animate, period, interval=600, blit=True, repeat=True)
    output.save('hail_per_month.gif', writer='imagemagick')
    
# draw animation for each day in year   
def draw_per_day():

    period = hail.day_month.unique()

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)

    m = Basemap(llcrnrlon=-130,llcrnrlat=13,urcrnrlon=-60,urcrnrlat=52,projection='merc')
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    m.shadedrelief()

    x,y = m(0, 0)
    point = m.plot(x, y, 'o', markersize=1, color='white')[0]

    
    def animate(i):
        p = hail[hail.day_month == i]

        fig.suptitle('Hail in {} day, {} month, 2015 year'.format(i[0:2], i[-1]), fontsize=22)
        lon = p.LON.values
        lat = p.LAT.values
        x, y = m(lon ,lat)
        point.set_data(x,y)
        return point,

    output = animation.FuncAnimation(plt.gcf(), animate, period, interval=100, blit=True, repeat=True)
    output.save('hail_per_day.gif', writer='imagemagick')
 
 
draw_per_month()
# draw_per_day()