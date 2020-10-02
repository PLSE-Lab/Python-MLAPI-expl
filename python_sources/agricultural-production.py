import pandas as pd
import numpy as np
import pylab as pl
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
from matplotlib import rcParams
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon, PathPatch


housea = pd.read_csv('../input/pums/ss13husa.csv')
data_a = {'PUMA': housea['PUMA'], 'ST':housea['ST'], 'AGS':housea['AGS']}
houseb = pd.read_csv('../input/pums/ss13husb.csv')
data_b = {'PUMA': houseb['PUMA'], 'ST':houseb['ST'], 'AGS':houseb['AGS']}

FrameA = pd.DataFrame(data_a)
house_frame_a = FrameA.dropna()
FrameB = pd.DataFrame(data_b)
house_frame_b = FrameB.dropna()



frames = [house_frame_a,house_frame_b]

agriProduction = pd.concat(frames)

agProductionByPUMA = agriProduction.pivot_table(index = ['PUMA', 'ST'], aggfunc = np.sum)
agProductionByState = agriProduction.pivot_table(index = ['ST'], aggfunc = np.sum)
agProductionByState= agProductionByState.drop('PUMA', 1)
print(agProductionByState.head())

agProductionByState.to_csv('states.csv')


fig = plt.figure()
ax = fig.add_subplot(111)
fig.suptitle('Food Stamp Recipients by PUMA', fontsize=20)

# create a map object with the Albert Equal Areas projection.
# This projection tends to look nice for the contiguous us.
m = Basemap(width=5000000,height=3500000,
            resolution='l',projection='aea',\
            lat_1=30.,lat_2=50,lon_0=-96,lat_0=38)

state_codes = {'01': 'Alabama',
               '02': 'Alaska',                               
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
               '72': 'Puerto Rico'
               }        
               




# define a color ramp

cm = plt.get_cmap('Blues')

categories = np.unique(agProductionByState['AGS'])

#add color bar legend
cmap = plt.set_cmap('Blues')

col_min = agProductionByState['AGS'].min()
col_max = agProductionByState['AGS'].max()


print(agProductionByState.AGS.max())
vmin= 0
vmax = 19288

agProductionByState= agProductionByState.sort(['AGS'], ascending = False)

agProductionByState['State'] = agProductionByState.index.tolist()

def stateCodeString(stateCode):
    if stateCode < 10:
        stateCode= "0" + str(stateCode)
        stateCode = state_codes[stateCode]
    else:
        stateCode= str(stateCode)
        stateCode = state_codes[stateCode]
    return stateCode


agProductionByState['State'] = agProductionByState.State.apply(stateCodeString)
agProductionByState.set_index(agProductionByState.State)
print(agProductionByState.head())

agProductionByState.plot(kind ='bar')
fig= plt.figure()
ax = fig.add_subplot(111)

agProductionByState.plot(kind ='bar', ax = ax)
plt.title('Agricultural Production by State')
ax.set_xticklabels(agProductionByState.State)
pl.show()
fig.savefig('figure1.png')
    


"""
#read each state's shapefile
for key in state_codes.keys():
    m.readshapefile('../input/shapefiles/pums/tl_2013_{0}_puma10'.format(key),
                    name='state', drawbounds=True)
    new_key = int(key)
    
    #loop through each PUMA and add it to plot
    for info, shape in zip(m.state_info, m.state):
        patches = [Polygon(np.array(shape), True)]
        pc = PatchCollection(patches, edgecolor = 'k', linewidths = 1., zorder = 2)
        ax.add_collection(pc)
        
    
        
        
        
m.pcolormesh(X, Y, agProductionByState['AGS'])
    
"""