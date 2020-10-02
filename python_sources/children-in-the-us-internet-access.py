'''
Examining the local ratio of children without Internet access in their households.

Author: Kati

Thanks to Phil Butcher for the demonstration of using shapefiles (https://www.kaggle.com/pjbutcher/2013-american-community-survey/making-a-map/code)


'''
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from mpl_toolkits.basemap import Basemap
import pandas as pd
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch

#PUMA; ST: state; HINCP: Household income; NRC: Number of children; NPF: number of people; ACCESS: access to Internet (3: No); BROADBND: mobile broadband (2 No)
housa=pd.read_csv("../input/pums/ss13husa.csv", usecols=['PUMA', 'ST', 'NRC', 'ACCESS'])
housb=pd.read_csv("../input/pums/ss13husb.csv", usecols=['PUMA', 'ST', 'NRC', 'ACCESS'])

data=pd.concat([housa,housb])

data=data.dropna(axis=0)
grouped=data.groupby(['ST','PUMA', 'ACCESS']).sum()

##There's probably prettier ways to do this part below...

#number of children without Net in each PUMA
noNet=data[data['ACCESS']==3].set_index(['ST','PUMA']).NRC
noNet=noNet.reset_index().groupby(['ST','PUMA']).sum().reset_index()
#total number of children in each PUMA
totalNum=data.groupby(['ST','PUMA']).NRC.sum().reset_index()
#percentage of children without Internet access in each PUMA
noNet['perc']=noNet['NRC']/totalNum['NRC']*100
noNet=noNet.groupby(['ST', 'PUMA'])['perc'].sum().reset_index()

#plotting
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


num=10
cm=plt.get_cmap('hot')
reds=[cm(1.0*i/num) for i in range(num-1,-1,-1)]
cmap = mpl.colors.ListedColormap(reds)

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, axisbg='w', frame_on=False)
fig.suptitle('Percentage of children without Internet access', fontsize=20)

m = Basemap(width=5000000,height=3500000,resolution='l',projection='aea',lat_1=30.,lat_2=50,lon_0=-96,lat_0=38)

for key in state_codes.keys():
    m.readshapefile('../input/shapefiles/pums/tl_2013_{0}_puma10'.format(key), name='state', drawbounds=True)
    new_key = int(key)
    

    for info, shape in zip(m.state_info, m.state):
        id=int(info['PUMACE10'])
        value=noNet[(noNet['ST']==new_key) & (noNet['PUMA']==id)]['perc']
        color=int(value/10)
        patches = [Polygon(np.array(shape), True)]
        pc = PatchCollection(patches, edgecolor='k', linewidths=1., zorder=2)
        pc.set_color(reds[color])
        ax.add_collection(pc)

ax2 = fig.add_axes([0.82, 0.1, 0.03, 0.8])
bounds=np.linspace(0,10,num)
cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, ticks=bounds, boundaries=bounds)
#,format='%1i')
cb.ax.set_yticklabels([str(round(i)*10) for i in bounds])

plt.show()
plt.savefig("children_without_net.png")
