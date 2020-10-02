'''
This script plots a heat map denoting the median age of houses in each state

Phil Butcher's method in map plotting is used in this script
Phil Butcher's original script:
https://www.kaggle.com/pjbutcher/2013-american-community-survey/making-a-map?tab=files/code
'''

import pandas as pd
import numpy as np

from pandas import DataFrame, Series

housing_a = '../input/pums/ss13husa.csv'
housing_b = '../input/pums/ss13husb.csv'

'''
YBL is the year when structure first built. 
Example:
01 .1939 or earlier
02 .1940 to 1949
03 .1950 to 1959
...
16 .2012
17 .2013
'''

husa = pd.read_csv("../input/pums/ss13husa.csv")
data_a = {'ST': husa['ST'], 'YBL':husa['YBL']}
husb = pd.read_csv("../input/pums/ss13husb.csv")
data_b = {'ST': husb['ST'], 'YBL':husb['YBL']}

h_data_a = DataFrame(data_a)
housing_data_a = h_data_a.dropna()
h_data_b = DataFrame(data_b)
housing_data_b = h_data_b.dropna()

housing = pd.concat([housing_data_a,housing_data_b], axis=0)

#Calculate the median for the year when house was first built
ST_median = housing.groupby('ST').median()

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
from matplotlib import rcParams
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon, PathPatch

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111)
fig.suptitle('The Median Lifespan of Housing by State', fontsize=20)

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
            #   '11': 'District of Columbia',                 
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
            #   '35': 'New Mexico',                           
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
               


# define a colorramp
num_colors = 8
cm = plt.get_cmap('Blues')
blues = [cm(1.*i/num_colors) for i in range(num_colors)]

# read each states shapefile
for key in state_codes.keys():
    m.readshapefile('../input/shapefiles/pums/tl_2013_{0}_puma10'.format(key),
                    name='state', drawbounds=True)
    new_key = int(key)
    
    #Assign color to each YBL (the year when house was first built)
    color_class = int(ST_median.ix[new_key])
    color = blues[color_class]
    
    # loop through each PUMA and assign a random color from our colorramp
    for info, shape in zip(m.state_info, m.state):
        patches = [Polygon(np.array(shape), True)]
        pc = PatchCollection(patches, edgecolor='k', linewidths=1., zorder=2)
        pc.set_color(color)
        ax.add_collection(pc)

plt.savefig('map.png')        