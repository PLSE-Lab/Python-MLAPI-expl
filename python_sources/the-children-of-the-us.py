#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# #The Children of the US
# ##Examining some aspects of the situation of children in the USA based on the 2013 census data
# 
# ###Introduction
# 
# Kaggle (www.kaggle.com) published the 2013 USA census data for its members to analyse and find some insights. 
# These data contain information about households and individuals all over the US. 
# I decided to analyse the economic structure of the country by examining the status of households with children.
# 
# ###1. Average number of children in households
# 
# First, let's take a look at how many children there are in an average household.

# In[ ]:


#loading packages
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from mpl_toolkits.basemap import Basemap
import pandas as pd
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch


# The datafiles have a columns telling us how many children are in that particular household

# In[ ]:


#read data
#ST: state; PUMA: district; NRC: number of children
housa=pd.read_csv("../input/pums/ss13husa.csv", usecols=['PUMA', 'ST', 'NRC'])
housb=pd.read_csv("../input/pums/ss13husb.csv", usecols=['PUMA', 'ST', 'NRC'])

data=pd.concat([housa,housb])
data=data.dropna(axis=0)


# Group the data by state and district

# In[ ]:


#group by state and district
grouped=data.groupby(['ST','PUMA'])


# Calculate average number of children per household for each district.

# In[ ]:


#calculate average
grouped=grouped['NRC'].agg([np.mean]).reset_index()
print ("Minimum value: {0: .3f}; maximum value: {1: .3f}, average: {2: .3f}, median: {3: .3f}".format(grouped['mean'].min(), grouped['mean'].max(), grouped['mean'].mean(), grouped['mean'].median()))


# The average number of children per household is between 0.128 and 1.592. The median is only 0.536, it is significantly lower than the value 
# necessary to keep the population number at least constant. 
# 
# Plot the results using the shapefiles for the districts that are provided with the data.

# In[ ]:


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
        color=int(value*10)
        patches = [Polygon(np.array(shape), True)]
        pc = PatchCollection(patches, edgecolor='k', linewidths=1., zorder=2)
        pc.set_color(reds[color])
        ax.add_collection(pc)
        label.append(color)
        


#add a legend
ax2 = fig.add_axes([0.82, 0.1, 0.03, 0.8])
bounds=np.linspace(0,1,num)
cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, ticks=bounds, boundaries=bounds)
#,format='%1i')
cb.ax.set_yticklabels([str(round(i*num)/10) for i in bounds])

plt.show()


# ###2. Is there a correlation between income of the household and the number of children?
# 
# Let's examine how much the number of children affect the income per person ni the US households.
# First we load the necessary columns of the data: the household income ("HINCP"), the number of children ("NRC") and the numberof children in the household.

# In[ ]:


housa=pd.read_csv("../input/pums/ss13husa.csv", usecols=['HINCP', 'NRC', 'NPF'])
housb=pd.read_csv("../input/pums/ss13husb.csv", usecols=['HINCP', 'NRC', 'NPF'])

data=pd.concat([housa,housb])
data=data.dropna(axis=0)


# I calculate the income/person value for each household.

# In[ ]:


data['INPP']=data['HINCP']/data['NPF']


# Since there are very few households with more than 10 children, we create a category to handle households with >10 children together. Then we group the data by the number of children.

# In[ ]:


data['NRC']=np.where(data['NRC']>10, 11, data['NRC'])
grouped=data.groupby('NRC')


# Now let's plot the data.

# In[ ]:


plt.figure(1)
income=grouped["INPP"].agg([np.mean,np.std])
income=income.reset_index()
labels=['%i'%j for j in range(11)]
labels.append(">10")
plt.xticks([j for j in range(12)],labels)
plt.axis([-0.5,11.5,-5000.0,85000.0])
plt.title("Mean income per person in households with different number of children")
plt.xlabel("Number of children")
plt.ylabel("Mean yearly income per person")
for i in range(len(income)):
    x=income["NRC"][i]
    y=income["mean"][i]
    yerror=income["std"][i]
    plt.scatter(x,y, color="black", s=20)
    plt.errorbar(x,y,yerr=yerror, color="black")
    
plt.show()


# Apparently there's a strong correlation between the number of children and the income/person in the household. The income/person decreases very fast with the number of children in households with less then 5 children, then seem to even out. The standard deviation, however, is large, so we should also look at the distribution of the data.
# In order to do this I plot the data on a boxplot.

# In[ ]:


bp=data.boxplot(column="INPP",by="NRC")
bp.get_figure().suptitle("")
plt.xticks([j for j in range(1,13)],labels)
plt.title("Distribution of income among housholds with  children")
plt.yticks([j for j in range(0,1100000,200000)],[j for j in range(0,12,2)])
plt.xlabel("Number of children")
plt.ylabel("Mean yearly income per person (100.000)")
plt.axis([-0.5,11.5,-10000.0,1100000.0])
plt.show()


# So apparently among households with no children there are much more people with really high incomes, and even with only one child it is true. However, the number of households with really high income decreases very quickly with the number of childen.
# 
# ###3. Do all children in the USA have access to the internet at home?
# 
# Nowadays having access to the internet is increasingly important. It is especially true for children, whose entire future can be determined based on their access to information. By examining what portion of children who has no access to internet at home we can study the status of them in different regions.
# Again, let's read the necessary data: the state ("ST"), the region ("PUMA"), the number of children in the household ("NRC") and whether or not they have access to the internet at home ("ACCESS").

# In[ ]:


housa=pd.read_csv("../input/pums/ss13husa.csv", usecols=['PUMA', 'ST', 'NRC', 'ACCESS'])
housb=pd.read_csv("../input/pums/ss13husb.csv", usecols=['PUMA', 'ST', 'NRC', 'ACCESS'])

data=pd.concat([housa,housb])
data=data.dropna(axis=0)


# Then I group the data by state and region and calculate the number of children with internet access with (ACCESS=1) and without subscription (ACCESS=2) and those without internet (ACCESS=3).

# In[ ]:


grouped=data.groupby(['ST','PUMA', 'ACCESS']).sum()


# n the next step I calculate the percentage of children without internet.

# In[ ]:


##There's probably prettier ways to do this part below...

#number of children without Net in each PUMA
noNet=data[data['ACCESS']==3].set_index(['ST','PUMA']).NRC
noNet=noNet.reset_index().groupby(['ST','PUMA']).sum().reset_index()
#total number of children in each PUMA
totalNum=data.groupby(['ST','PUMA']).NRC.sum().reset_index()
#percentage of children without Internet access in each PUMA
noNet['perc']=noNet['NRC']/totalNum['NRC']*100
noNet=noNet.groupby(['ST', 'PUMA'])['perc'].sum().reset_index()


# Then I plot the data.

# In[ ]:


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
cb.ax.set_yticklabels([str(round(i)*10) for i in bounds])

plt.show()


# Though in many parts of the US the percentage of children without net is under 10 per cent, there are mant areas with 10-30 percent ratio, and, surprisingly there are regions where more than 60 per cent of the children have no internet access at home!
