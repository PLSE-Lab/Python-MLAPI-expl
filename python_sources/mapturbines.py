# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

#%% Get the data
wtfile='../input/windTurbines.csv'
wtdat=pd.read_csv(wtfile)

#% setup the map
fig,ax=plt.subplots(figsize=(20,10))

c1=(-127.6+66.1)/2
c2=(50.5+23.2)/2
m = Basemap(resolution='i', # c, l, i, h, f or None
            projection='merc',
            lat_0=c2, lon_0=c1,
            llcrnrlon=-127.6, llcrnrlat= 23.2, urcrnrlon=-66.1, urcrnrlat=50.5)

# draw the map
m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color='#f2f2f2',lake_color='#46bcec')
m.drawcoastlines(linewidth=0.5)
m.drawcountries(linewidth=0.5)
m.drawstates(linewidth=0.5)
x,y=m(np.array(wtdat['long_DD']),np.array(wtdat['lat_DD']))

# plot each turbine with marker size based on machine size

mx=wtdat['MW_turbine'].max()
for i in range(len(wtdat)):
    if wtdat['MW_turbine'].loc[i]>0:
        m.plot(x[i],y[i],'o',
            color='red',
            markersize=wtdat['MW_turbine'].loc[i]/mx*15,
            alpha=0.25,
            markeredgecolor='none')
    else:
        m.plot(x[i],y[i],
            'x',
            color='brown')

plt.show()    
fig.savefig('WindTurbineMap.png',orientation='landscape')