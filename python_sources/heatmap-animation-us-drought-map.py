#########################3#
### Written by Jaeyoon Park 
### 16 Jan 2017
### Python 3.5
###########################

#########################
### PART 1: DATA HANDLING
#########################

import pandas as pd
import numpy as np

### Read in county data and drought info data
county = pd.read_csv('../input/county_info_2016.csv', encoding = "ISO-8859-1")
county.columns = ['USPS','GEOID','ANSICODE','NAME','ALAND','AWATER','ALAND_SQMI','AWATER_SQMI','INTPTLAT','INTPTLONG' ]
county = county[['GEOID','ALAND_SQMI','INTPTLAT','INTPTLONG']]

dr = pd.read_csv('../input/us-droughts.csv')

### Check if there are NaNs and all entries are recorded week's internal
#dr.isnull().sum() ## To check if there are NaNs
#dr.validStart = pd.to_datetime(dr.validStart, format='%Y-%m-%d')
#dr.validEnd = pd.to_datetime(dr.validEnd, format='%Y-%m-%d')
#((dr.validEnd - dr.validStart)!="6 day").sum()
#dr.releaseDate = pd.to_datetime(dr.releaseDate, format='%Y-%m-%d')
#(dr.releaseDate != dr.validStart).sum()

### Data cleansing
dr.releaseDate = pd.to_datetime(dr.releaseDate, format='%Y-%m-%d')
num_m = len(dr.releaseDate.unique())
dr = dr.drop(dr[['county', 'state', 'validStart', 'validEnd', 'domStatisticFormatID']], axis=1)

### Resample data on monthly basis
dr_m = dr.set_index('releaseDate').groupby(['FIPS']).resample('M').mean()

### Calculate drought level (when NONE is 100% => 0, if D4 is 100% => 5, and linearly between 0 and 5)
dr_m['LEVEL'] = (dr_m.D4*5 + (dr_m.D3-dr_m.D4)*4 + (dr_m.D2-dr_m.D3)*3 + (dr_m.D1-dr_m.D2)*2 + (dr_m.D0-dr_m.D1))/100

### Merge drought data and county info (coordination, size) by FIPS/GEOID
dr_m = dr_m.reset_index(level=1)
dr_final = pd.merge(dr_m, county, left_on='FIPS', right_on='GEOID', how='inner', sort='False')
dr_final = dr_final[['FIPS', 'releaseDate', 'LEVEL', 'ALAND_SQMI', 'INTPTLAT', 'INTPTLONG']]
dr_final = dr_final.groupby('FIPS')


#####################################
### PART 2: HEATMAP ANIMATION DISPLAY
#####################################

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from mpl_toolkits.basemap import Basemap
from matplotlib.animation import FuncAnimation
#get_ipython().magic('matplotlib nbagg') ## Only for Jupyter iPython display

### Create a figure and draw a basemap
fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111)
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
m.drawcoastlines() #zorder=3
m.drawmapboundary(zorder=0) #fill_color='#9fdbff'
m.fillcontinents(color='#ffffff',zorder=1) #,lake_color='#9fdbff',alpha=1
m.drawcountries(linewidth=1.5) #color='darkblue'
m.drawstates() #zorder=3

### Set county location values, drough level values, marker sizes (according to county size), colormap and title 
x, y = m(dr_final.nth(0).INTPTLONG.tolist(), dr_final.nth(0).INTPTLAT.tolist())
colors = (dr_final.nth(0).LEVEL).tolist()
sizes = (dr_final.nth(0).ALAND_SQMI/7.5).tolist()
cmap = plt.cm.YlOrRd
sm = ScalarMappable(cmap=cmap)
plt.title('US Drought Level (Year-Month): '+dr_final.nth(0).releaseDate.iloc[0].strftime('%Y-%m'))

### Display the scatter plot and its colorbar (0-5)
scatter = ax.scatter(x,y,s=sizes,c=colors,cmap=cmap,alpha=1,edgecolors='face',marker='H',vmax=5,vmin=0,zorder=1.5)
plt.colorbar(scatter)

## Update function for animation
def update(ii):
    colors = (dr_final.nth(ii).LEVEL).tolist()
    scatter.set_color(sm.to_rgba(colors))
    plt.title('US Drought Level (Year-Month): '+dr_final.nth(ii).releaseDate.iloc[0].strftime('%Y-%m'))
    return scatter, 

anim = FuncAnimation(plt.gcf(),update,interval=300,repeat=False,frames=203,blit=True) #blit=True
anim.save('Heatmap_animation_US_Drought.gif', writer='imagemagick')
#plt.show()

###################################
### PART 3: ANIMATION SAVING AS MP4
###################################

### Save functions for animation 
#from matplotlib import rc, animation
#mywriter = animation.FFMpegWriter()
#anim.save('Heatmap_animation_US_Drought.mp4',writer=mywriter)

