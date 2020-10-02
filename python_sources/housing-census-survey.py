import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch

#read data
housea = pd.read_csv("../input/pums/ss13husa.csv", usecols=['PUMA', 'ST', 'RNTP'])
houseb = pd.read_csv("../input/pums/ss13husb.csv", usecols=['PUMA', 'ST', 'RNTP'])

#concat tables
data = pd.concat([housea, houseb])
data = data.dropna(axis=0)

#group by state and district
grouped=data.groupby(['ST','PUMA'])

#calculate average
grouped=grouped['RNTP'].agg([np.mean]).reset_index()
print ("Minimum value: {0: .3f}; maximum value: {1: .3f}, average: {2: .3f}, median: {3: .3f}".format(grouped['mean'].min(), grouped['mean'].max(), grouped['mean'].mean(), grouped['mean'].median()))
print (grouped)

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

#colormap
num=int(round(grouped['mean'].max()*10))
cm=plt.get_cmap('hot')
reds=[cm(1.0*i/num) for i in range(num-1,-1,-1)]
cmap = mpl.colors.ListedColormap(reds)

#set up figure
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, axisbg='w', frame_on=False)
fig.suptitle('Average number of children per household', fontsize=20)

#set up map
m = Basemap(width=5000000,height=3500000,resolution='l',projection='aea',lat_1=30.,lat_2=50,lon_0=-96,lat_0=38)
label=[]
#loop through the states
for key in state_codes.keys():
    m.readshapefile('../input/shapefiles/pums/tl_2013_{0}_puma10'.format(key), name='state', drawbounds=True)
    new_key = int(key)
    

    for info, shape in zip(m.state_info, m.state):
        id=int(info['PUMACE10'])
        value=grouped[(grouped['ST']==new_key) & (grouped['PUMA']==id)]['mean']
        color=int(value)
        patches = [Polygon(np.array(shape), True)]
        pc = PatchCollection(patches, edgecolor='k', linewidths=1., zorder=2)
        pc.set_color(reds[color])
        ax.add_collection(pc)
        label.append(color)
        


#add a legend
ax2 = fig.add_axes([0.82, 0.1, 0.03, 0.8])
#bounds=np.linspace(0,1,num)
#cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, ticks=bounds, boundaries=bounds)
#,format='%1i')
#cb.ax.set_yticklabels([str(round(i*num)/10) for i in bounds])

plt.show()