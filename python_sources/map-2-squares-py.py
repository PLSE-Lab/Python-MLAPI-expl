### 
### output a map with two squares, one over ocean, one over land.
### 

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np

m = Basemap(projection='merc',
            resolution='l',lon_0=-120.,
            llcrnrlon=-140, llcrnrlat=20,
            urcrnrlon=-55, urcrnrlat=50)
m.shadedrelief()

### ocean square
dolat1 = 37.5
dolon1 = -127.5

### land square
dolat2 = 37.5
dolon2 = -117.5

lats1 = [dolat1-2.5, dolat1-2.5, dolat1+2.5, dolat1+2.5, dolat1-2.5]
lons1 = [dolon1-2.5, dolon1+2.5, dolon1+2.5, dolon1-2.5, dolon1-2.5]

lats2 = [dolat2-2.5, dolat2-2.5, dolat2+2.5, dolat2+2.5, dolat2-2.5]
lons2 = [dolon2-2.5, dolon2+2.5, dolon2+2.5, dolon2-2.5, dolon2-2.5]

x1, y1 = m(lons1, lats1)
x2, y2 = m(lons2, lats2)

m.plot(x1, y1, marker=None, color='b', linewidth=2)
m.plot(x2, y2, marker=None, color='g', linewidth=2)

### add some annotation
x1, y1 = m(dolon1, dolat1)
x2, y2 = m(dolon1, dolat1-10)
label1 = 'Ocean\nsquare'
plt.annotate(label1, xy=(x1, y1),  xycoords='data',
             xytext=(x2, y2), textcoords='data',
             arrowprops=dict(arrowstyle="->"), size='large'
            )

x1, y1 = m(dolon2,    dolat2)
x2, y2 = m(dolon2+10, dolat2)
label2 = 'Land\nsquare'
plt.annotate(label2, xy=(x1, y1),  xycoords='data',
             xytext=(x2, y2), textcoords='data',
             arrowprops=dict(arrowstyle="->"), size='large'
            )

plt.savefig('plot0_map_ocean_land.png', bbox_inches="tight")
plt.show()
