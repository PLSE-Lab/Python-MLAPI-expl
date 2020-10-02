#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train=pd.read_csv("/kaggle/input/forest-train/train.csv")


# In[ ]:


train.head()


# In[ ]:


train.columns


# In[ ]:


train.head()


# In[ ]:


#Rename the wilderness area columns.
train.rename(columns = {'Wilderness_Area1':'Rawah', 'Wilderness_Area2':'Neota','Wilderness_Area3':'Comanche_Peak ','Wilderness_Area4':'Cache_la_Poudre'}, inplace = True) 


# In[ ]:


train.columns


# In[ ]:


train['Wild_area'] = (train.iloc[:, 10:15] == 1).idxmax(1)
#Combining the four wilderness area columns
train['Soil_type'] = (train.iloc[:, 15:55] == 1).idxmax(1)
#combining fourty soil type columns
# Removing the already existing ones
train_forest=train.drop(columns=['Rawah', 'Neota',
       'Comanche_Peak ', 'Cache_la_Poudre', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4',
       'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9',
       'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13',
       'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 'Soil_Type17',
       'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21',
       'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25',
       'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29',
       'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33',
       'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37',
       'Soil_Type38', 'Soil_Type39', 'Soil_Type40'])


# In[ ]:


train_forest.describe()


# # Data analysis

# In[ ]:


unique, counts = np.unique(train.Cover_Type, return_counts=True)
(unique, counts) # equal number of Type


# In[ ]:


fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
train[['Cover_Type', 'Elevation']].groupby(['Cover_Type']).mean().plot(kind='bar', ax=ax1, color='k')
train[['Cover_Type', 'Aspect']].groupby(['Cover_Type']).mean().plot(kind='bar', ax=ax2, color='b')
train[['Cover_Type', 'Slope']].groupby(['Cover_Type']).mean().plot(kind='bar', ax=ax3, color='r')


# In[ ]:


label=['Cover ' + str(x) for x in range(1,8)]
for i in range(7):
    ax = plt.hist(train.Elevation[train.Cover_Type==i+1],label=label[i], bins=20,stacked=True)
plt.legend()
plt.xlabel('Elevation (m)')


# In[ ]:


# Elevation feature has an important weight.
#since we can already distinguish the type 3, 4 and 7 with only this attribute.


# # The wilderness areas 
# 
# 1 - Rawah Wilderness Area
# 2 - Neota Wilderness Area
# 3 - Comanche Peak Wilderness Area
# 4 - Cache la Poudre Wilderness Area

# In[ ]:


#entries from different wilderness areas.
sns.countplot(train_forest['Wild_area']);

# Comanche Peak wild Area occur the most and Neota occurs least


# In[ ]:


# number of entries of each cover type.
sns.countplot(train_forest['Cover_Type'],palette="rainbow");

#shows same frequency


# In[ ]:


#Distribution of aspect values:
sns.distplot(train_forest['Aspect'],kde=False,color='blue', bins=100);
plt.ylabel('Frequency',fontsize=10)

#most of the values lying between 0-100 and 275-350.


# In[ ]:


#Distribution of elevation values:
sns.distplot(train_forest['Elevation'],kde=False,color='pink', bins=100);
plt.ylabel('Frequency',fontsize=10)


# In[ ]:


#data collected from relatively high altitude areas
#peaks at the intervals of 2000m-2500m, 2500m-3000m,3000m-3500m and tapering at both ends.


# In[ ]:


#Distribution of values of slope
sns.distplot(train_forest['Slope'],kde=False,color='maroon', bins=100);
plt.ylabel('Frequency',fontsize=10)

#Right skewed(positive), highest frequency around 10 at the slope


# # Horizontal distance

# In[ ]:


#Distribution of values of the horizontal distance to roadways.
sns.distplot(train_forest['Horizontal_Distance_To_Roadways'],kde=False,color='green', bins=100);
plt.ylabel('Frequency',fontsize=10)


# In[ ]:


#most of the samples are within the range of 0-2000 distance to the roadways.
#This indicates chances of commercial exploitation of the forests by human activities.


# In[ ]:


#Distribution of values of the horizontal distance to fire points.
sns.distplot(train_forest['Horizontal_Distance_To_Fire_Points'],kde=False,color='red', bins=100);
plt.ylabel('Frequency',fontsize=10)


# In[ ]:


#This positive skewed distribution indicates the horizontal distance to wildfire ignition point mostly around 0-2000
#thus indicates some influence of human activities in the proximities


# In[ ]:


#Distribution of values of the horizontal distance to nearest surface water features.
sns.distplot(train_forest['Horizontal_Distance_To_Hydrology'],kde=False,color='aqua', bins=100);
plt.ylabel('Frequency',fontsize=10)


# In[ ]:


#peak near zero: This means that most of the samples are present very close to water sources.


# # Vertical distance

# In[ ]:


#Distribution of values of the vertical distances to nearest surface water features.
sns.distplot(train_forest['Vertical_Distance_To_Hydrology'],kde=False,color='blue', bins=100);
plt.ylabel('Frequency',fontsize=10)

# shows a positively skewed distribution with a sharp peak at 0.


# # Hillshade index

# In[ ]:


#Distribution of values of the hillshade index at 9am during summer solstice on an index from 0-255.
sns.distplot(train_forest['Hillshade_9am'],kde=False,color='orange', bins=100);
plt.ylabel('Frequency',fontsize=10)

# shows a negtively skewed distribution, peaking at around 225 in the range 100 to 250.


# In[ ]:


sns.distplot(train_forest['Hillshade_Noon'],kde=False,color='orange', bins=100);
plt.ylabel('Frequency',fontsize=10)

#negtively skewed distribution, peak around 225 in the range 125-250.


# In[ ]:


sns.distplot(train_forest['Hillshade_3pm'],kde=False,color='orange', bins=100);
plt.ylabel('Frequency',fontsize=10)

# shows a more or less symmetric distribution


# In[ ]:



color = ['b','r','k','y','m','c','g']
for i in range(7):
    plt.scatter(train.Hillshade_Noon[train.Cover_Type==i+1], train.Hillshade_3pm[train.Cover_Type==i+1], color=color[i], label='Type' +str(i+1))
plt.xlabel('Hillshade_Noon')
plt.ylabel('Hillshade_3pm')
plt.legend()


# In[ ]:


#Shows a scatterplot where several values are equal to 0 which may be the missing values and replaced with mean of the attribute.


# # Analysis of Forest cover type

# The seven forest cover types are:
# 
# 1 - Spruce/Fir
# 2 - Lodgepole Pine
# 3 - Ponderosa Pine
# 4 - Cottonwood/Willow
# 5 - Aspen
# 6 - Douglas-fir
# 7 - Krummholz

# In[ ]:


sns.countplot(data=train_forest,x='Wild_area',hue="Cover_Type");


# #Rawah wilde area has forest cover types: Spruce/Fir,Lodgepole Pine,Aspen and Krummholz.
# #Forest cover type Lodgepole Pine is present max in Rawah and min in Cache la poudre wild area.
# #Comanche peak wild area has all the forest cover types, except Neota wild area, which has only 3 forest cover types:     Spruce/Fir,Lodgepole Pine and Krummholz. They represent two extremes.
# #The forest cover type Cottonwood/Willow is a major in the Cache la Poudre wild area.
# 

# In[ ]:


#Boxplot between elevation and Cover type
sns.boxplot(y=train['Elevation'],x=train['Cover_Type'],palette='rainbow')


# #This box plot shows distribution of forest cover types based on elevation.
# Most of them are at similar elavations, except for forest cover type Krummholz(7), present at the highest median elavation of around 3375m; followed by type 1.
# #Forest cover type Lodgepole Pine(2);Aspen(5) and Douglas-fir(6);Ponderosa Pine(3) occur at same elavations. 
# #The forest cover type Cottonwood/Willow(4) shows the least median elavation.

# In[ ]:


#Boxplot between slope and Cover type
sns.boxplot(y=train['Slope'],x=train['Cover_Type'],palette='rainbow')


# #we can see that the box plots overlap with Slight variations in median slope

# In[ ]:


#Boxplot between Aspect and Cover type
sns.boxplot(y=train['Aspect'],x=train['Cover_Type'],palette='rainbow')


# #there is no significant variations in the median values as they all lie within the range 100-200.

# In[ ]:


#Creating data frame for Degree Variables 
train_deg=train[['Elevation','Aspect','Slope','Cover_Type']]
sns.pairplot(train_deg,hue='Cover_Type')


# #For 'Aspect' and 'Slope' each forest cover type has almost equal distribution.
# 
# #So, we can say 'Elevation' can play a role in classification

# In[ ]:


#Boxplot between Horizontal_Distance_To_Roadways and Cover type
sns.boxplot(y=train['Horizontal_Distance_To_Roadways'],x=train['Cover_Type'],palette='nipy_spectral')


# ##The median varies from 1000-2000 across all the cover types except 7.
# ##most types are equidistant from roadways, which indicates human impacts.
# 

# In[ ]:


#Boxplot between Horizontal_Distance_To_Hydrology and Cover type
sns.boxplot(y=train['Horizontal_Distance_To_Hydrology'],x=train['Cover_Type'],palette='nipy_spectral')


# #the median lies in the range 0-300
# ##Cottonwood/Willow(4)) is present close to water source.
# ##Krummholz(7)) is present most distant to water source.

# In[ ]:


#Boxplot between Vertical_Distance_To_Hydrology and Cover type
sns.boxplot(y=train['Vertical_Distance_To_Hydrology'],x=train['Cover_Type'],palette='nipy_spectral')


# #we see that the values do not vary much among the different cover types.
# #This implies that it is not an important factor in further analysis.

# In[ ]:


#Boxplot between Horizontal_Distance_To_Fire_Points and Cover type
sns.boxplot(y=train['Horizontal_Distance_To_Fire_Points'],x=train['Cover_Type'],palette='nipy_spectral')


# #It shows least for type 3,4 and 6.
# ##implies that all types are vulnerable to forest fires as they are closer to ignition points.

# In[ ]:


#Creating data frame for Distance Variables 
train_dist=train[['Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways',
          'Horizontal_Distance_To_Fire_Points','Cover_Type']]

#Creating pairplot for Degree Variables
sns.pairplot(train_dist,hue='Cover_Type')


# #From the above plots, it is evident that distances are playing major role in classification of forest cover type

# In[ ]:


#Boxplot between Hillshade_Noon and Cover type
sns.boxplot(y=train['Hillshade_Noon'],x=train['Cover_Type'],palette='plasma')


# In[ ]:


#All values are lying between 220 to 240.


# In[ ]:


#Boxplot between Hillshade_3pm and Cover type
sns.boxplot(y=train['Hillshade_3pm'],x=train['Cover_Type'],palette='rainbow')


# In[ ]:


#values are ranging between 100 to 150.


# In[ ]:


#Boxplot between Hillshade_3pm and Cover type
sns.boxplot(y=train['Hillshade_9am'],x=train['Cover_Type'],palette='rainbow')


# In[ ]:


#values are in similar range of around 200 to 250.


# In[ ]:


#Creating data frame for Hillshade Variables 
train_hs=train[['Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Cover_Type']]
#Creating pairplot for Hillshade Variables
sns.pairplot(train_hs,hue='Cover_Type')


# #'Hillshade_9am' and 'Hillshade_Noon' have differnt ranges of start index for all forest cover types .
# 
# #Where as , 'Hillshade_3pm' gives almost same ranges for all forest cover type.

# # Analysis of Wilderness area

# In[ ]:


#Elevation values across wilderness areas and forest cover types.
sns.catplot(data=train_forest,x='Wild_area',y="Elevation",hue="Cover_Type");


# ##It is clear that the forest cover type Krummholz(7)) has highest elevation point, present in Comanche Peak region eventhough Neota area has the highest elevation.
# #Cottonwood/willow(4)) tree is present in low elevation region.
# 

# In[ ]:


#Horizontal distance to fire points across wilderness areas and forest cover types.
sns.catplot(data=train_forest,x='Wild_area',y="Horizontal_Distance_To_Fire_Points",hue="Cover_Type");


# #we can understand that in Rawah area, cover type Lodgepole Pine(2) is far from ignition point,present in large frequency.
# 
# #In Comanche peak, Krummholz(7) is far from firepoint while in Neota area this cover type is closer to firepoint.
# 
# 

# In[ ]:


#Relationship between wild areas and distance to roadways across forest cover types.
sns.catplot(data=train_forest,x='Wild_area',y="Horizontal_Distance_To_Roadways",hue="Cover_Type");


# #Krummholz(7) forest cover type is in general far from roadways.
# 
# #Roadways are important areas of human interaction and movement. So, it would be logical to see the relationship between distance to roadways and firepoint.

# In[ ]:


#Relationship between distance to firepoints and roadways across forest cover types.
sns.lmplot(data=train_forest,x='Horizontal_Distance_To_Fire_Points',y='Horizontal_Distance_To_Roadways',scatter=False,hue="Cover_Type");


# #This graph explains the exception of Cache la Poudre area. 
# #Cottonwood/Willow(4) shows neagtive correlation between firepoint and roadways distances.
# 
# #we can say that horizontal distance to fire points and horizontal distance are directly proportional.
# #Cottonwood/Willow forest cover type and Cache la Poudre area does not follow the normal trend

# # Analysis of Soil type

# In[ ]:


# plotting Soil types based on forest cover types.
sns.catplot(y="Soil_type",hue="Cover_Type",kind="count",palette="viridis",height=15,data=train_forest);


# #This graph shows the soil types in which the seven forest cover types are present:
# 
# #Spruce/Fir and Lodgepole Pine : highest frequency in soil type 29 i.e, Como - Legault families complex, extremely stony.
# #Ponderosa Pine, Douglas-fir: highest frequency in soil type 10; Bullwark - Catamount families - Rock outcrop complex, rubbly.
# #Cottonwood/Willow: highest frequency in soil type 3; Haploborolis - Rock outcrop complex, rubbly.
# #Aspen: highest frequency in soil type 30.
# #Krummholz: highest frequency in soil type 38,Leighcan - Moran families - Cryaquolls complex, extremely stony.

# In[ ]:


# plotting Soil types based on Wilderness areas.
sns.catplot(y="Soil_type",hue="Wild_area",kind="count",palette="plasma",height=15,data=train_forest);


# #This plot shows the frequencies of soil types found in the various wilderness areas:
# #Cache la pourde :highest frequency of soil type 10- Bullwark as well as in types 3 and 6(Haploborolis,Vanet)
# #Rawah: highest frequencies of soil types 29 and 30- como family
# #Comanche peak: simultaneously highest frequencies of soil types 32,10 and 4- Catamount ,Bullwark and Ratake .
# #Neota: comparitively lower frequencies of soil types 38 and 40-Leighcan and Moran family.
# 
# 

# In[ ]:


# Heatmap 
size = 10
mat = dataset_train.iloc[:,:size].corr()
f, ax = plt.subplots(figsize = (10,8))
sns.heatmap(mat,vmax=0.8,square=True);


# #This is a correlation matrix or heatmap that requires continuous data. This does not include wilderness area and soil type.
# 

# In[ ]:





# In[ ]:





# In[ ]:




