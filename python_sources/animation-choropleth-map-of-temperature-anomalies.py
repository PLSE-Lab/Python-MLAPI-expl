
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from mpl_toolkits.basemap import Basemap, cm
from matplotlib import animation
import matplotlib as mpl


climat = pd.read_csv('../input/ghcn-m-v1.csv')

#Prepare data
east = []
west = []
for x in climat.columns:
    if 'E' in x:
        east.append(x)
    elif 'W' in x:
        west.append(x)
east = dict(zip(east,[x+2.5 for x in range(0,180,5)]))
west = dict(zip(west,[x+2.5 for x in range(-180,0,5)]))

north = []
south = []
for x in climat.lat.unique():
    if 'N' in x:
        north.append(x)
    elif 'S' in x:
        south.append(x)
north = dict(zip(north,[x+2.5 for x in reversed(range(0,90,5))]))
south = dict(zip(south,[x+2.5 for x in reversed(range(-115,0,5))]))

climat.lat.replace(north, inplace=True)
climat.lat.replace(south, inplace=True)
climat.rename(columns=west, inplace=True)
climat.rename(columns=east, inplace=True)

new = pd.DataFrame(columns=('year', 'month','lat', 'temp','lon'))
for x in climat.columns[3:]:
    df = climat[['year', 'month','lat', x]][climat[x] != -9999]
    new = pd.concat([new, pd.DataFrame({'year': df.year.values, 'month':df.month.values, 'lat':df.lat.values, 'temp':df[x].values, 'lon':[x]*len(df)})])

new.temp = new.temp /100
new.year = new.year.astype(np.int)
new.month = new.month.astype(np.int)
new.info()


#Let's make animation :)

# add new features for indexing data
new['year_month'] = new.year.apply(lambda x: str(x)) + new.month.apply(lambda x: str(x)) 

# set the desired period
period = new[new.year >=2014].year_month.unique()


# make our animation
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111)

m = Basemap(projection='merc',
                     llcrnrlat=-80,
                     urcrnrlat=80,
                     llcrnrlon=-180,
                     urcrnrlon=180,
                     lat_ts=0,
                     resolution='l')
m.drawcoastlines()
m.drawcountries()
m.fillcontinents(color = 'gainsboro',alpha= 0.4)
m.drawmapboundary(fill_color='steelblue')

# draw parallels and meridians.
# labels = [left,right,top,bottom]
parallels = np.arange(-90.,90.,5.)
m.drawparallels(parallels,labels=[True,False,False,False])
meridians = np.arange(0.,360.,20.)
m.drawmeridians(meridians,labels=[True,False,True,False])

x,y = m([],[])
point = m.plot(x, y,)[0]


# make our animation
def animate(i):
    p = new[new.year_month == i]

    num_colors = 10
    values = p.temp.values
    cm = plt.get_cmap('Oranges')
    scheme = [cm(i / num_colors) for i in range(num_colors)]
    bins = np.linspace(values.min(), values.max(), num_colors)
    p['bin'] = np.digitize(values, bins) - 1
    p.sort_values('bin', ascending=False).head(10)

    fig.suptitle('Tempature in month {}, year {}'.format(i[-1], i[0:4]), fontsize=22 )
    lon = p['lon'].values
    lat = p['lat'].values
    x, y = m(lon ,lat)
    ax.scatter(x,y, s = 250, marker='s', c = [scheme[p[(p.lat ==x) & (p.lon==y)]['bin'].values] for x,y in zip(p.lat, p.lon)])

    # Draw color legend.
    ax_legend = fig.add_axes([0.16, 0.06, 0.7, 0.03], zorder=3)
    cmap = mpl.colors.ListedColormap(scheme)
    cb = mpl.colorbar.ColorbarBase(ax_legend, cmap=cmap, ticks=bins, boundaries=bins, orientation='horizontal')
    cb.ax.set_xticklabels([str(round(i, 1)) for i in bins])
    return point, 

output = animation.FuncAnimation(plt.gcf(), animate, period, interval=150, blit=True, repeat=True)
output.save('temperature.gif', writer='imagemagick')