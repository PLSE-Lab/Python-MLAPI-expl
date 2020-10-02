#!/usr/bin/env python
# coding: utf-8

#  The study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado. Each observation is a 30m x 30m patch. You are asked to predict an integer classification for the forest cover type.
# The wilderness areas are:
# 
# 1 - Rawah Wilderness Area
# 2 - Neota Wilderness Area
# 3 - Comanche Peak Wilderness Area
# 4 - Cache la Poudre Wilderness Area

# The seven types are:
# 
# 1 - Spruce/Fir
# 2 - Lodgepole Pine
# 3 - Ponderosa Pine
# 4 - Cottonwood/Willow
# 5 - Aspen
# 6 - Douglas-fir
# 7 - Krummholz

# Data fields:-
# 
# Elevation - Elevation in meters
# Aspect - Aspect in degrees azimuth
# Slope - Slope in degrees
# Horizontal_Distance_To_Hydrology - Horz Dist to nearest surface water features
# Vertical_Distance_To_Hydrology - Vert Dist to nearest surface water features
# Horizontal_Distance_To_Roadways - Horz Dist to nearest roadway
# Hillshade_9am (0 to 255 index) - Hillshade index at 9am, summer solstice
# Hillshade_Noon (0 to 255 index) - Hillshade index at noon, summer solstice
# Hillshade_3pm (0 to 255 index) - Hillshade index at 3pm, summer solstice
# Horizontal_Distance_To_Fire_Points - Horz Dist to nearest wildfire ignition points
# Wilderness_Area (4 binary columns, 0 = absence or 1 = presence) - Wilderness area designation
# Soil_Type (40 binary columns, 0 = absence or 1 = presence) - Soil Type designation
# Cover_Type (7 types, integers 1 to 7) - Forest Cover Type designation

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


#read the train.csv file into a data frame called df. 
df=pd.read_csv("/kaggle/input/train.csv")


# In[ ]:


df


# In[ ]:


#Renaming the wilderness area columns for better clarity.
df.rename(columns = {'Wilderness_Area1':'Rawah', 'Wilderness_Area2':'Neota','Wilderness_Area3':'Comanche_Peak','Wilderness_Area4':'Cache_la_Poudre'}, inplace = True) 


# In[ ]:


df.columns


# In[ ]:


df.info()


# In[ ]:


#Combining the four wilderness area columns and fourty soil type columns to Wild_area and Soil_type respectively, and removing already existing ones.
df['Wild_area'] = (df.iloc[:, 11:15] == 1).idxmax(1)
df['Soil_type'] = (df.iloc[:, 15:55] == 1).idxmax(1)
df=df.drop(columns=['Id','Rawah', 'Neota',
       'Comanche_Peak', 'Cache_la_Poudre', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4',
       'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9',
       'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13',
       'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 'Soil_Type17',
       'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21',
       'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25',
       'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29',
       'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33',
       'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37',
       'Soil_Type38', 'Soil_Type39', 'Soil_Type40'])
df


# In[ ]:


df.describe()


# In[ ]:


plt.figure(figsize=(15,15))
plt.subplots_adjust(hspace=.5)

plt.subplot(3,2,1)
plt.hist(df['Elevation'],color="red")
plt.xlabel('Elevation')

plt.subplot(3,2,2)
plt.hist(df['Slope'],color="blue")
plt.xlabel('Slope')

plt.subplot(3,2,3)
plt.hist(df['Horizontal_Distance_To_Hydrology'],color="green")
plt.xlabel('Horizontal_Distance_To_Hydrology')

plt.subplot(3,2,4)
plt.hist(df['Horizontal_Distance_To_Roadways'],color="yellow")
plt.xlabel('Horizontal_Distance_To_Roadways')

plt.subplot(3,2,5)
plt.hist(df['Horizontal_Distance_To_Fire_Points'],color="violet")
plt.xlabel('Horizontal_Distance_To_Fire_Points')

plt.subplot(3,2,6)
plt.hist(df['Aspect'],color="orange")
plt.xlabel('Aspect')


# Total Roosevelt National Forest
# 
# The average elevtion is at 2749m with value sranging from 1863m to 3849m.
# 
# The mean horizontal distance to surface water features is 227 units and for vertical distance 51 units.
# 
# Mean horizontal distance to roadways is 1714 units. The values range from 0 units to 6890 units and hence the standard deviation is 1325 units.
# 
# Mean horizontal distance to firepoints is 1511 units. The values range from 0 to 6993 units.
# 
# Mean slope is 16.5 units. The values range from 0 to 52.

# Comparing data fields in each wilderness area.

# In[ ]:


Rawah=df.loc[df.Wild_area == 'Rawah',:]
Rawah.describe()


# In[ ]:


plt.figure(figsize=(15,15))
plt.subplots_adjust(hspace=.5)
plt.suptitle('RAWAH WILD AREA')

plt.subplot(3,2,1)
plt.hist(Rawah['Elevation'],color="red")
plt.xlabel('Elevation')

plt.subplot(3,2,2)
plt.hist(Rawah['Slope'],color="blue")
plt.xlabel('Slope')

plt.subplot(3,2,3)
plt.hist(Rawah['Horizontal_Distance_To_Hydrology'],color="green")
plt.xlabel('Horizontal_Distance_To_Hydrology')

plt.subplot(3,2,4)
plt.hist(Rawah['Horizontal_Distance_To_Roadways'],color="yellow")
plt.xlabel('Horizontal_Distance_To_Roadways')

plt.subplot(3,2,5)
plt.hist(Rawah['Horizontal_Distance_To_Fire_Points'],color="violet")
plt.xlabel('Horizontal_Distance_To_Fire_Points')

plt.subplot(3,2,6)
plt.hist(Rawah['Aspect'],color="orange")
plt.xlabel('Aspect')


# Rawah Wilderness Area
# 
# The average elevtion is at 2996m with values ranging from 2482m to 3675m.
# 
# The mean horizontal distance to surface water features is 223.6 units and for vertical distance 38 units.
# 
# Mean horizontal distance to roadways is 2586 units. The values range from 67 units to 6890 units and hence the standard deviation is 1802.9 units.
# 
# Mean horizontal distance to firepoints is 2359 units. The values range from 42 to 6993 units.
# 
# Mean slope is 14.1 units. The values range from 0 to 50.

# In[ ]:


Neota=df.loc[df.Wild_area == 'Neota',:]
Neota.describe()


# In[ ]:


plt.figure(figsize=(15,15))
plt.subplots_adjust(hspace=.5)
plt.suptitle('Neota Wilderness Area')

plt.subplot(3,2,1)
plt.hist(Neota['Elevation'],color="red")
plt.xlabel('Elevation')

plt.subplot(3,2,2)
plt.hist(Neota['Slope'],color="blue")
plt.xlabel('Slope')

plt.subplot(3,2,3)
plt.hist(Neota['Horizontal_Distance_To_Hydrology'],color="green")
plt.xlabel('Horizontal_Distance_To_Hydrology')

plt.subplot(3,2,4)
plt.hist(Neota['Horizontal_Distance_To_Roadways'],color="yellow")
plt.xlabel('Horizontal_Distance_To_Roadways')

plt.subplot(3,2,5)
plt.hist(Neota['Horizontal_Distance_To_Fire_Points'],color="violet")
plt.xlabel('Horizontal_Distance_To_Fire_Points')

plt.subplot(3,2,6)
plt.hist(Neota['Aspect'],color="orange")
plt.xlabel('Aspect')


# Neota Wilderness Area
# 
# 
# The average elevtion is at 3341m with values ranging from 2978m to 3643m.
# 
# The mean horizontal distance to surface water features is 326.6 units and for vertical distance 56.7 units.
# 
# Mean horizontal distance to roadways is 1134.79 units. The values range from 108 units to 2505 units and hence the standard deviation is 511.7 units.
# 
# Mean horizontal distance to firepoints is 1777.7 units. The values range from 30 to 5753 units.
# 
# Mean slope is 13.48 units. The values range from 1 to 39.
# 

# In[ ]:


Comanche=df.loc[df.Wild_area == 'Comanche_Peak',:]
Comanche.describe()


# In[ ]:


plt.figure(figsize=(15,15))
plt.subplots_adjust(hspace=.5)
plt.suptitle('Comanche Peak Wilderness Area')

plt.subplot(3,2,1)
plt.hist(Comanche['Elevation'],color="red")
plt.xlabel('Elevation')

plt.subplot(3,2,2)
plt.hist(Comanche['Slope'],color="blue")
plt.xlabel('Slope')

plt.subplot(3,2,3)
plt.hist(Comanche['Horizontal_Distance_To_Hydrology'],color="green")
plt.xlabel('Horizontal_Distance_To_Hydrology')

plt.subplot(3,2,4)
plt.hist(Comanche['Horizontal_Distance_To_Roadways'],color="yellow")
plt.xlabel('Horizontal_Distance_To_Roadways')

plt.subplot(3,2,5)
plt.hist(Comanche['Horizontal_Distance_To_Fire_Points'],color="violet")
plt.xlabel('Horizontal_Distance_To_Fire_Points')

plt.subplot(3,2,6)
plt.hist(Comanche['Aspect'],color="orange")
plt.xlabel('Aspect')


# Comanche Peak Wilderness Area
# 
# The average elevtion is at 2923m with values ranging from 2301m to 3849m.
# 
# The mean horizontal distance to surface water features is 276.7 units and for vertical distance 56 units.
# 
# Mean horizontal distance to roadways is 1907.72 units. The values range from 0 units to 5463 units and hence the standard deviation is 1072.2 units.
# 
# Mean horizontal distance to firepoints is 1518.2 units. The values range from 30 to 4481 units.
# 
# Mean slope is 15.37 units. The values range from 0 to 52.
# 

# In[ ]:


cache=df.loc[df.Wild_area == 'Cache_la_Poudre',:]
cache.describe()


# In[ ]:


plt.figure(figsize=(15,15))
plt.subplots_adjust(hspace=.5)
plt.suptitle('Cache la Poudre Wilderness Area')
plt.subplot(3,2,1)
plt.hist(cache['Elevation'],color="red")
plt.xlabel('Elevation')

plt.subplot(3,2,2)
plt.hist(cache['Slope'],color="blue")
plt.xlabel('Slope')

plt.subplot(3,2,3)
plt.hist(cache['Horizontal_Distance_To_Hydrology'],color="green")
plt.xlabel('Horizontal_Distance_To_Hydrology')

plt.subplot(3,2,4)
plt.hist(cache['Horizontal_Distance_To_Roadways'],color="yellow")
plt.xlabel('Horizontal_Distance_To_Roadways')

plt.subplot(3,2,5)
plt.hist(cache['Horizontal_Distance_To_Fire_Points'],color="violet")
plt.xlabel('Horizontal_Distance_To_Fire_Points')

plt.subplot(3,2,6)
plt.hist(cache['Aspect'],color="orange")
plt.xlabel('Aspect')


# Cache la Poudre Wilderness Area
# 
# The average elevtion is at 2260m with values ranging from 1863m to 2622m.
# 
# The mean horizontal distance to surface water features is 152.05 units and for vertical distance 53.5 units.
# 
# Mean horizontal distance to roadways is 841.6 units. The values range from 0 units to 1770 units and hence the standard deviation is 394.69 units.
# 
# Mean horizontal distance to firepoints is 820.68 units. The values range from 0 to 2001 units.
# 
# Mean slope is 20.12 units. The values range from 1 to 47.

# covertype vs other variables 

# In[ ]:


sns.catplot(y="Soil_type",hue="Wild_area",kind="count",palette="pastel",height=10,data=df);


# In[ ]:


sns.boxplot(y=df['Elevation'],x=df['Cover_Type'],palette="husl");


# In[ ]:


sns.boxplot(y=df['Aspect'],x=df['Cover_Type'],palette='husl');


# In[ ]:


sns.boxplot(y=df['Slope'],x=df['Cover_Type'],palette='husl');


# In[ ]:


sns.boxplot(y=df['Horizontal_Distance_To_Hydrology'],x=df['Cover_Type'],palette='husl');


# In[ ]:


sns.boxplot(y=df['Vertical_Distance_To_Hydrology'],x=df['Cover_Type'],palette='husl');


# In[ ]:


sns.boxplot(y=df['Horizontal_Distance_To_Roadways'],x=df['Cover_Type'],palette='husl');


# In[ ]:


sns.boxplot(y=df['Hillshade_9am'],x=df['Cover_Type'],palette='husl');


# In[ ]:


sns.boxplot(y=df['Hillshade_Noon'],x=df['Cover_Type'],palette='husl');


# In[ ]:


sns.boxplot(y=df['Hillshade_3pm'],x=df['Cover_Type'],palette='husl');


# In[ ]:


sns.boxplot(y=df['Horizontal_Distance_To_Fire_Points'],x=df['Cover_Type'],palette='husl');


# In[ ]:


p1=sns.countplot(data=df,x='Wild_area',hue="Cover_Type",palette='husl')
p1.set_xticklabels(p1.get_xticklabels(),rotation=15);


# In[ ]:


a=df[['Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points','Cover_Type','Wild_area']]
sns.pairplot(a,hue='Wild_area')


# In[ ]:


b=df[['Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points','Cover_Type','Wild_area']]
sns.pairplot(b,hue='Cover_Type')


# In[ ]:


sns.lmplot(data=df,x='Horizontal_Distance_To_Hydrology',y='Elevation',scatter=False,hue="Cover_Type")


# ## Please Upvote my notebook :)
