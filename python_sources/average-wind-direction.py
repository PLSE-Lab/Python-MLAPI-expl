import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('../input/CLIWOC15.csv')
es_dict = pd.read_csv('../input/Lookup_ES_WindDirection.csv')
fr_dict = pd.read_csv('../input/Lookup_FR_WindDirection.csv')
nl_dict = pd.read_csv('../input/Lookup_NL_WindDirection.csv')
uk_dict = pd.read_csv('../input/Lookup_UK_WindDirection.csv')

wind_dict = {}
for i in es_dict.values: wind_dict[i[1]] = i[2]
for i in fr_dict.values: wind_dict[i[1]] = i[2]
for i in nl_dict.values: wind_dict[i[1]] = i[2]
for i in uk_dict.values: wind_dict[i[1]] = i[2]

lat = data.Lat3
lon = data.Lon3
wind = data.WindDirection

#collect list of windpoints
wind_points = []
import matplotlib.colors as colors
import matplotlib.cm as cmx
DATA = np.random.rand(5,5)
cmap = plt.cm.jet
cNorm  = colors.Normalize(vmin=np.min(DATA[:,4]), vmax=np.max(DATA[:,4]))

scalarMap = cmx.ScalarMappable(norm=cNorm,cmap=cmap)
check = 75000
errors = 0
coords_n_data = list(zip(lat,lon,wind))
for entry in coords_n_data[:check]:
    if not np.isnan(entry[0]) and not np.isnan(entry[1]):
        try:
            wind_direction = wind_dict[entry[2]]
            wind_points.append((entry[0],entry[1],wind_direction))
            #colorVal = scalarMap.to_rgba(wind_direction/360.0000)
            #print colorVal, wind_direction
            #plt.scatter(entry[0], entry[1], s=5, alpha=.2, c=colorVal)
        except:
            errors+=1
a,b,c = zip(*wind_points)
x_range = int(min(a)), int(max(a))
y_range = int(min(b)), int(max(b))
increment = 1
x = x_range[0]
y = y_range[0]
import matplotlib.pyplot as plt
import math
while x < x_range[1]:
    while y < y_range[1]:
        #print x,y, 'location being chekced'
        #plt.scatter(x,y,s=4,c='r')
        directions = []
        points_plot = []
        for p in wind_points:
            if int(p[0])==x and int(p[1])==y:
                directions.append(p[2])
                points_plot.append(p)
        for i in points_plot: wind_points.remove(i)    
        if len(directions)>5:
            d=np.average(directions)
            colorVal = scalarMap.to_rgba(d/360.0000)
            plt.scatter(x, y, s=30, alpha=.4, c=colorVal)
        y+= increment
    y = y_range[0]
    x += increment
plt.savefig('wind_map.png', figsize=(10,10), dpi=200)
#plt.show()
