#!/usr/bin/env python
# coding: utf-8

# The study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado. Each observation is a 30m x 30m patch. You are asked to predict an integer classification for the forest cover type. The seven types are:
# 
# 1 - Spruce/Fir
# 2 - Lodgepole Pine
# 3 - Ponderosa Pine
# 4 - Cottonwood/Willow
# 5 - Aspen
# 6 - Douglas-fir
# 7 - Krummholz
# 
# The wilderness areas are:
# 
# 1 - Rawah Wilderness Area
# 2 - Neota Wilderness Area
# 3 - Comanche Peak Wilderness Area
# 4 - Cache la Poudre Wilderness Area 

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


os.chdir('../input')


# In[ ]:


train= pd.read_csv('train.csv')


# In[ ]:


train.head(2)


# In[ ]:


train.shape


# In[ ]:


train.columns


# In[ ]:


train.rename(columns= {'Wilderness_Area1': 'Rawah', 'Wilderness_Area2': 'Neota', 'Wilderness_Area3': 'Comanche Peak', 'Wilderness_Area4':'Cache la Poudre'}, inplace= True)


# In[ ]:


train.columns


# In[ ]:


#Combining the four wilderness area columns and fourty soil type columns to Wild_area and Soil_type respectively, and removing already existing ones.
train['Wild'] = (train.iloc[:, 11:15] == 1).idxmax(1)
train['Soil'] = (train.iloc[:, 15:55] == 1).idxmax(1)
train13 = train.drop(columns=['Id','Rawah', 'Neota',
       'Comanche Peak', 'Cache la Poudre', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4',
       'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9',
       'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13',
       'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 'Soil_Type17',
       'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21',
       'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25',
       'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29',
       'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33',
       'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37',
       'Soil_Type38', 'Soil_Type39', 'Soil_Type40'])
train13


# In[ ]:


train13.describe() 


# This descriptive statistical values gives an idea of the critical values required for analysis.
# 
# ==>The average elevtion is around 2750m with value ranging from 1863m to 3849m.
# 
# ==>The mean horizontal distance to surface water features is 227 units and for vertical distance 51 units.
# 
# ==>Mean horizontal distance to roadways is 1714 units. The values range from 0 units to 6890 units and hence the standard deviation is 1325 units.
# 
# ==>Mean horizontal distance to firepoints is 1511 units. The values range from 0 to 6993 units.

# In[ ]:





# In[ ]:


sns.countplot(train13['Cover_Type'], color= 'silver') 
#gives the number of entries in each cover type
# ==> all cover types appear to have the same frequency


# In[ ]:


sns.countplot(train13['Wild'], palette= 'spring');  


# Data entries from Comanche Peak wild area is more frequent while the entries from Neota wild area is very small. It is also clear that the entries from all 4 wild area have different frequency of occurence. Although the samples from each cover type is same, the sampling within each wilderness area must have been irregular.

# In[ ]:





# In[ ]:





# In[ ]:


sns.scatterplot(train13['Elevation'], train13['Aspect'], hue= train13['Cover_Type'], palette= 'plasma')  
# ==> relationship between differnet variables and cover type


# In[ ]:


sns.scatterplot(train13['Vertical_Distance_To_Hydrology'], train13['Horizontal_Distance_To_Hydrology'], hue= train13['Cover_Type'], palette= 'seismic')


# In[ ]:





# In[ ]:


sns.boxplot(x= train13['Cover_Type'], y= train13['Elevation'], palette= 'magma') 


# Although the highest range of both cover type 1 and 7 seems to have the same elevation point, cover type 7 has higher median at around 3400m, and likewise, while cover type has the lowest range of elevation, cover type 4 has the lowest elevation median at 2250m. 

# In[ ]:


sns.boxplot(x= train13['Cover_Type'], y= train13['Slope'], palette= 'rainbow')  


# Cover type 1 and 7 has the lowest slope median value at around 13m,cover type 3 has the highest slope median value at around 22m. But the values of slope do not vary very significantly between various forest cover types, as it can be observed that the box plots overlap. 

# In[ ]:


sns.boxplot(x= train13['Cover_Type'], y= train13['Aspect'],notch= True, palette= 'nipy_spectral') 


# In[ ]:


ax= sns.boxplot(x= train13['Cover_Type'], y= train13['Aspect'],palette= 'nipy_spectral') 
ax= sns.swarmplot(x= train13['Cover_Type'], y= train13['Aspect'],color= 'silver') 


# Cover type 5 has the lowest aspect median value at 120 and cover type 5 and 6 has the highest aspect median value at 170. Cover type 4 has the lowest range, while all other cover types have frwquencies at the same range.
# And from the swarmplot, we can see that the data collected for all cover types are equal.

# In[ ]:


sns.boxplot(x= train13['Cover_Type'], y= train13['Wild'], palette= 'Dark2') 


# 

# In[ ]:


tt= pd.read_csv('train.csv')


# In[ ]:


sns.boxplot(x= tt['Cover_Type'], y= tt['Wilderness_Area1'], palette= 'Dark2') 


# To get a clearer picture, plotting cover type against Wilderness area 1. ==> cover type 3,4,6 seems to be have the lowest range in Wilderness area 1, i.e., Rawah., while the other cover types are evenly spread.

# In[ ]:


sns.boxplot(x= tt['Cover_Type'], y= tt['Wilderness_Area2'], palette= 'Dark2') 


# Wilderness area 2, i.e., Comanche Peak, seems to have outliers in cover type 1,2,7

# In[ ]:


sns.boxplot(x= tt['Cover_Type'], y= tt['Wilderness_Area3'], palette= 'Dark2') 


# Cover type 4 looks to have the lowest range in wilderness area 3 or Cache la Poudre.

# In[ ]:


sns.boxplot(x= tt['Cover_Type'], y= tt['Wilderness_Area4'], palette= 'Dark2') 


# Cover type 1,2,5,7 seems to have the lowest range and cover type 3 and 4 are uniformly distributed in Wilderness area 4 or Neota.

# In[ ]:





# In[ ]:


sns.catplot(y= 'Wild', hue= 'Cover_Type', kind= 'count', palette='viridis', height= 15, data=train13) 


# This graph showes the wild types in which the seven forest cover types are present.
# ==> cover type 2/ Lodgepole Pine is present in the highest frequency in Rawah.
# ==> cover type 7/Krummholz is present in the highest frequency in Comanche peak.
# ==> cover type 3/Ponderosa Pine is present in the highest frequency in Cache la Poudre.
# ==> cover type 7/Krummholz is present in the highest frequency in Neota.

# In[ ]:


X= train13[['Elevation', 'Slope', 'Aspect', 'Cover_Type']] 


# In[ ]:


sns.pairplot(X, hue= 'Cover_Type', palette= 'magma') 


# Cover types 1 and 7 elevation value lies between 2500 and 4000m and for forest cover type 2 elevation value lies between 2000-3500m.
# For Aspect and Slope each forest cover type has almost equal distribution. So it can be said that Elevation may play a role in classification.

# In[ ]:


X1= train13[['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points', 'Cover_Type']] 


# In[ ]:


sns.pairplot(X1, hue= 'Cover_Type', palette= 'spring') 


# For cover type '3','4' and '6' horizontal distance to hydrology are not going to upper values and for '1','2' ,'5' and '7', it's going to higher range.

# In[ ]:


sns.boxplot(x= 'Cover_Type', y= 'Horizontal_Distance_To_Hydrology', palette= 'rainbow', data= train13)


# Overall, the median of the distance vary between 0-200
# Cover type1(Spruce/Fir) is the closest to water source while type7(Krummholz) is the most distant.

# In[ ]:


sns.boxplot(x= 'Cover_Type', y= 'Horizontal_Distance_To_Roadways', palette= 'rainbow', data= train13)


# Type5(Aspen) is the closest to road while type7(Krummholz) is the farthest from road. This could imply that the trees are prone to human interaction and exploitation.

# In[ ]:


sns.boxplot(x= 'Cover_Type', y= 'Horizontal_Distance_To_Fire_Points', palette= 'rainbow', data= train13) 


# The median horizontal distance to firepoints varies from 1000-2000 units across the various forest cover types. Therefore, on an average, all cover types could be prone to forest fire. The distance to roadways seems to be in the similar range as the distance to firepoints, therefore it can be said that they might influence each other.

# In[ ]:


sns.boxplot(x= 'Cover_Type', y= 'Vertical_Distance_To_Hydrology', palette= 'summer', data= train13)


# The vertical distance seems to come in the same range for all cover types.

# Looking at other variables to find more about the wilderness area...

# In[ ]:


sns.boxplot(x= 'Wild', y= 'Elevation', data= train13, palette= 'gist_earth')


# Neota area has the highest elevation. Cache la Poudre has the lowest elevation. Rawah and Comanche areas has similar elevations.

# In[ ]:


sns.catplot(y = 'Elevation',x= 'Wild',hue= 'Cover_Type', data=train13) 


# This graph further confirms the fact that Cache la Poudre has low elevation since covertype7(Krummholz) is absent here and covertype4(Cottonwood/Willow) is present only here. Although the elevation median value of Neota was higher, it can be noted that Comanche has the highest elevation point. Therefore, it can be inferred that elevation influence the type of vegetation in the respective areas.

# In[ ]:


sns.boxplot(x= 'Wild', y= 'Horizontal_Distance_To_Hydrology', data= train13, palette= 'ocean')


# Cache la poudre is the closest to water source, and neota the farthest but all are close to water source in one way or the other.

# In[ ]:


sns.catplot(y = 'Horizontal_Distance_To_Hydrology',x= 'Wild',hue= 'Cover_Type', data=train13) 


# Cover type1(Spruce/Fir) is the closest to water source while type7(Krummholz) is the most distant.

# In[ ]:


sns.boxplot(x= 'Wild', y= 'Horizontal_Distance_To_Roadways', data= train13, palette= 'ocean')


# Rawah area is farthest from roadways, Cache la Poudre is closer to roadways.

# In[ ]:


sns.catplot(y= 'Horizontal_Distance_To_Roadways', x= 'Wild', hue= 'Cover_Type', data= train13)


# It can be seen that covertype7(Krummholz) forest cover type is far from roadways except in the case of Neota. This could mean human interactions, since the roadways are close to the forests.

# In[ ]:


sns.boxplot(x= 'Wild',y= 'Horizontal_Distance_To_Fire_Points',data= train13, palette= 'ocean')


# Cache la Poudre is more closer to firepoints and thus more prone to forest fire while Rawah is farther away from the firepoints and therefore comparatively less prone to forest fire.

# In[ ]:


sns.catplot(x= 'Wild', y= 'Horizontal_Distance_To_Fire_Points', hue= "Cover_Type", data= train13)


# In Rawah area, covertype2 is farthest from firepoint and covertype5and 1 is close to firepoint.
# In Comanche area, covertype3,6 is close and covertype7 is fathest from firepoint.
# All of Cache la Poudre is close to firepoint. In Neota area, covertype1 is far and covertype7 is close to firepoint.

# Looking at the relationship of roadways and firepoints with respect to elevation.

# In[ ]:


sns.lmplot(x= 'Elevation', y='Horizontal_Distance_To_Roadways', scatter= False, data= train13);


# In[ ]:


sns.lmplot(x='Elevation',y='Horizontal_Distance_To_Fire_Points',scatter=False, data= train13);


# To look at the relationship between roadways and firepoints..

# In[ ]:


sns.lmplot(x='Horizontal_Distance_To_Fire_Points',y='Horizontal_Distance_To_Roadways',scatter=False, data= train13);


# Hence, from the three graphs, it can be inferred that as elevation value increases, the distance to roadways and firepoints also increases.

# To see how the relationship between roadways and firepoints are effected with respect to wilderness type.

# In[ ]:


sns.lmplot(x='Horizontal_Distance_To_Fire_Points',y='Horizontal_Distance_To_Roadways',hue="Wild", scatter=False,data=train13, palette= 'ocean')


# From the above plot, it can be inferred that except in the case of Cache la Poudre, in all others the distance to roadways and firepoints are directly proportional.

# To further explain this..

# In[ ]:


sns.lmplot(x='Horizontal_Distance_To_Fire_Points',y='Horizontal_Distance_To_Roadways',hue="Cover_Type",scatter=False,data= train13, palette= 'winter');


# This graph further validates the exception of Cache la Poudre, which is because of the presence of covertype4(Cottonwood/Willow) is present only in this region and it shows neagtive correlation between firepoint and roadways distances. It must be remembered that Cache la Poudre area is closest to roadways and firepoints.

# Mostly,it can be said that horizontal distance to fire points and horizontal distance are directly proportional.
# This might also be due to human influence on fire point.
# Covertype4(Cottonwood/Willow) forest cover type and consequently Cache la Poudre area does not follow the normal trend even though this area is closest to roadways and firepoints,which might be because this forest type might be innately highly inflammable and does not require human interference for starting a fire.

# In[ ]:





# In[ ]:


sns.boxplot(y= train13['Horizontal_Distance_To_Hydrology'],x= train13['Wild'], palette= 'viridis_r');


# All wild areas are more or less close to water bodies.

# In[ ]:


sns.catplot(x='Wild',y="Horizontal_Distance_To_Hydrology",hue="Cover_Type",data= train13, palette= 'viridis_r');


# In[ ]:


sns.lmplot(x='Elevation',y='Horizontal_Distance_To_Hydrology',data= train13,scatter=False);


# In[ ]:


sns.boxplot(y= train13['Vertical_Distance_To_Hydrology'],x= train13['Wild'], palette= 'viridis');


# In[ ]:


sns.catplot(x='Wild',y="Vertical_Distance_To_Hydrology",hue="Cover_Type", data= train13, palette= 'viridis');


# In[ ]:





# In[ ]:


X2= train[['Hillshade_9am', 'Hillshade_Noon',
       'Hillshade_3pm', 'Cover_Type']]


# In[ ]:


sns.pairplot(X2, hue= 'Cover_Type', palette= 'winter')


# Hillshade_9am and Hillshade_Noon have different ranges of start index for all forest cover types while Hillshade_3pm gives almost same ranges for all forest cover type.

# Looking deeper into the pairplot..

# In[ ]:


sns.boxplot(x= 'Cover_Type', y= 'Hillshade_9am', palette= 'cividis', data= train13)


# The hillshade9am median values for all cover types lies between range 200-250.

# In[ ]:


sns.boxplot(x= 'Cover_Type', y= 'Hillshade_Noon', palette= 'cividis', data= train13)


# The hillshade noon median values for all cover types lie between range 220-240.

# In[ ]:


sns.boxplot(x= 'Cover_Type', y= 'Hillshade_3pm', palette= 'cividis', data= train13)


# The hillshade3pm median value lie between the range of 100-150.

# From the above three graphs, it can be inferred that the hillshade values of 9am and noon lie in similar range while hillshade3pm value lies in a lower range. All the cover types lie in a similar range irrespective of the hillshade time. 

# In[ ]:


sns.boxplot(x= train13['Wild'],y= train13['Hillshade_9am'], palette= 'cool');


# In[ ]:


sns.boxplot(x= train13['Wild'],y= train13['Hillshade_Noon'], palette= 'cool');


# In[ ]:


sns.boxplot(x= train13['Wild'],y= train13['Hillshade_3pm'], palette= 'cool');


# In[ ]:


sns.boxplot(x= train13['Wild'],y= train13['Slope'], palette= 'cool');


# In[ ]:


sns.boxplot(x= train13['Wild'],y= train13['Aspect'], palette= 'cool');


# Since from the above graphs, these features are found to not vary much with forest cover type,further analysis might not be required.

# In[ ]:


sns.catplot(y= 'Soil', hue= 'Cover_Type', kind= 'count', palette='viridis_r', height= 15, data=train13) 


# This graph showes the soil types in which the seven forest cover types are present.
# 
# ==> 1(Spruce/Fir) is present in the highest frequency in soil type 29(Como - Legault families complex, extremely stony).
# 
# ==> 2(Lodgepole Pine) is present in the highest frequency in soil type 29(Como - Legault families complex, extremely stony).
# 
# ==> 3(Ponderosa Pine) is present in the highest frequency in soil type 10(Bullwark - Catamount families - Rock outcrop complex, rubbly).
# 
# ==> 4(Cottonwood/Willow) is present in the highest frequency in soil type 3( Haploborolis - Rock outcrop complex, rubbly).
# 
# ==> 5(Aspen) is present in the highest frequency in soil type 30(Como family - Rock land - Legault family complex, extremely stony).
# 
# ==> 6(Douglas-fir) is present in the highest frequency in soil type 10(Bullwark - Catamount families - Rock outcrop complex, rubbly).
# 
# ==> 7(Krummholz) is present in the highest frequency in soil type 38(Leighcan - Moran families - Cryaquolls complex, extremely stony).

# In[ ]:


sns.catplot(y="Soil",hue="Wild",kind="count",palette="Dark2_r",height=10,data= train13);


# In[ ]:





# From the analysis, the following conclusion can be made on cover types..

# 1 - Spruce/Fir..
# Present in all wilderness areas except Cache la Pourde, it is present in the highest frequency in the Rawah wilderness area.
# It occurs at a relatively high elevation of around 3125m, second only to Krummholz forest type.
# It is present in the highest frequency in soil type 29(Como - Legault families complex, extremely stony), but is present in other soil types as well.
# It is present relatively farther from fire points and roadways. This could mean that this is a high altitude tree that has less human interference.
# It is not very far from surface water sources, but not very close either.

# 2 - Lodgepole Pine..
# Present in all wilderness although predominantly present in Rawah. Not very close to water sources or firepoints. Close to roadways in Cache la Poudre and Neota area.Elevation range between 2500-3250. Present in highest frequency in soil type 29(Como - Legault families complex, extremely stony).

# 3 - Ponderosa Pine..Present only in Comanche and Cache la Paudre. Elevation value highest at 2750m. Present in highest frequency in soil type 10(Bullwark - Catamount families - Rock outcrop complex, rubbly). Close to water sources. Also close to roadways and firepoints, therefore prone to human interfernces and forest fire.

# 4 - Cottonwood/Willow.. Present only in Cache la Poudre. Elevation value highest at around 2350m. Present in highest frequency in soil type 3( Haploborolis - Rock outcrop complex, rubbly). Close to water sources. Also close to roadways and firepoints, therefore prone to human interfernces and forest fire.

# 5 - Aspen..Present in Rawah and Comanche, although predominantly in Rawah. Elevation value highest at around 2900m. Present in high frequency in soil type 30(Como family - Rock land - Legault family complex, extremely stony). Close to water bodies, raodways and firepoints.

# 6 - Douglas-fir..Present only in Comanche peak and Cache la Poudre. Elevation value highest at around 2900m. Present in high frequency in soil type 10(Bullwark - Catamount families - Rock outcrop complex, rubbly). Close to water bodies, raodways and firepoints.

# 7 - Krummholz..Present in all wild areas except Cache la Paudre.It has the highest elevation value above 3750m. Present in high frequency in soil type 38(Leighcan - Moran families - Cryaquolls complex, extremely stony). It is far from roadways except in area Neota. It is most distant to water sources. It is closer to firepoint in the case of Neota and further away in case Rawah and Comanche peaks

# In[ ]:





# In[ ]:




