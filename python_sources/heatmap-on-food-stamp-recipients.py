'''
This script plots a heat map denoting the median age of houses in each state

Phil Butcher's method in map plotting is used in this script
Phil Butcher's original script:
https://www.kaggle.com/pjbutcher/2013-american-community-survey/making-a-map?tab=files/code

WenruLiu's method adapted
WenruLiu's original script:
https://www.kaggle.com/wenruliu/2013-american-community-survey/is-there-a-young-city

still working on this 
'''

import pandas as pd
import numpy as np

from pandas import DataFrame, Series

housing_a = '../input/pums/ss13husa.csv'
housing_b = '../input/pums/ss13husb.csv'

'''
FS do the occupants recieve foodstamps
b. vacant (N/A)
1. Yes
2. No
'''

husa = pd.read_csv("../input/pums/ss13husa.csv")
data_a = {'PUMA': husa['PUMA'],'ST': husa['ST'], 'FS':husa['FS']}
husb = pd.read_csv("../input/pums/ss13husb.csv")
data_b = {'PUMA': husb['PUMA'],'ST': husb['ST'], 'FS':husb['FS']}

h_data_a = DataFrame(data_a)
housing_data_a = h_data_a.dropna()
h_data_b = DataFrame(data_b)
housing_data_b = h_data_b.dropna()

foodstamps = pd.concat([housing_data_a,housing_data_b], axis=0)


foodstampsY = foodstamps[foodstamps.FS ==1]
foodstampsY.to_csv('foostampdata.csv')

#foodstampsY.drop(['ST'])
#Pivot table by PUMA

PUMAFS = foodstampsY.pivot_table(index= ['PUMA'], aggfunc = np.sum)







import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
from matplotlib import rcParams
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon, PathPatch

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111)
fig.suptitle('Food Stamp Recipients by PUMA', fontsize=20)

# create a map object with the Albert Equal Areas projection.
# This projection tends to look nice for the contiguous us.
m = Basemap(width=5000000,height=3500000,
            resolution='l',projection='aea',\
            lat_1=30.,lat_2=50,lon_0=-96,lat_0=38)

state_codes = {'01': 'Alabama',
            #   '02': 'Alaska',                               
               '04': 'Arizona',                              
               '05': 'Arkansas',                             
               '06': 'California',                           
               '08': 'Colorado',                             
               '09': 'Connecticut',                          
               '10': 'Delaware',                            
               '11': 'District of Columbia',                 
               '12': 'Florida',                              
               '13': 'Georgia',                              
               '15': 'Hawaii',                               
               '16': 'Idaho',                                
               '17': 'Illinois',                             
               '18': 'Indiana',                              
               '19': 'Iowa',
               '20': 'Kansas',                               
               '21': 'Kentucky',                             
               '22': 'Louisiana',                            
               '23': 'Maine',                                
               '24': 'Maryland',                             
               '25': 'Massachusetts',                        
               '26': 'Michigan',                         
               '27': 'Minnesota',                            
               '28': 'Mississippi',                          
               '29': 'Missouri',                           
               '30': 'Montana',                              
               '31': 'Nebraska',                             
               '32': 'Nevada',                              
               '33': 'New Hampshire',                        
               '34': 'New Jersey',                         
               '35': 'New Mexico',                           
               '36': 'New York',                             
               '37': 'North Carolina',                       
               '38': 'North Dakota',                         
               '39': 'Ohio',                                 
               '40': 'Oklahoma',                             
               '41': 'Oregon',                              
               '42': 'Pennsylvania',                         
               '44': 'Rhode Island',                         
               '45': 'South Carolina',                       
               '46': 'South Dakota',                         
               '47': 'Tennessee',                            
               '48': 'Texas',                                
               '49': 'Utah',                                 
               '50': 'Vermont',                              
               '51': 'Virginia',                             
               '53': 'Washington',                           
               '54': 'West Virginia',                        
               '55': 'Wisconsin',                            
               '56': 'Wyoming',                              
            #   '72': 'Puerto Rico'
               }        
               


# define a color ramp
num_colors = 12
cm = plt.get_cmap('Blues')
blues = [cm(1.*i/num_colors) for i in range(num_colors)]

#add color bar legend
cmap = mpl.colors.ListedColormap(blues)


#define color numbers which determine shade of color
bounds = np.linspace(1, 12, num_colors)
#map bins to a color
PUMAFS['color_class'] = pd.cut(PUMAFS.FS, 12, labels = bounds)



PUMAFS.to_csv('foodstampinfo.csv')

print(PUMAFS.dtypes)
print(PUMAFS.head())
print(PUMAFS.FS.loc[100])

# read each states shapefile
for key in state_codes.keys():
    m.readshapefile('../input/shapefiles/pums/tl_2013_{0}_puma10'.format(key),
                    name='state', drawbounds=True)
    new_key = int(key)
    

    
    # loop through each PUMA and assign a random color from our colorramp
    for info, shape in zip(m.state_info, m.state):
        color = int(PUMAFS.color_class.loc[int(info['PUMACE10'])])%12
        patches = [Polygon(np.array(shape), True)]
        pc = PatchCollection(patches, edgecolor='k', linewidths=1., zorder=2)
        pc.set_color(blues[color])
        ax.add_collection(pc)
        
#second axis for color bar
ax2 = fig.add_axes([0.82, 0.1, 0.03, 0.8])
cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, ticks=bounds, boundaries=bounds,
                               format='%1i')
cb.ax.set_yticklabels([str(round(i, 2)) for i in bounds])

plt.savefig('map.png')        

