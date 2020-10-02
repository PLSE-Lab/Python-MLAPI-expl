#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)


# The study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado.
# The seven forest cover types are:
# 
# 1 - Spruce/Fir
# 2 - Lodgepole Pine
# 3 - Ponderosa Pine
# 4 - Cottonwood/Willow
# 5 - Aspen
# 6 - Douglas-fir
# 7 - Krummholz

# The wilderness areas are:
# 
# 1 - Rawah Wilderness Area
# 2 - Neota Wilderness Area
# 3 - Comanche Peak Wilderness Area
# 4 - Cache la Poudre Wilderness Area

# In[ ]:


os.chdir("../input/")


# In[ ]:


forest=pd.read_csv('eda.csv')


# In[ ]:


forest.head(10)


# In[ ]:


forest.dtypes


# In[ ]:


forest.columns


# In[ ]:


forest_new=forest.drop(['Id'],axis=1)
forest_new.head()


# In[ ]:


forest_new.shape


# In[ ]:


print(forest_new.isnull().sum())


# In[ ]:


# Therefore its clear from the above data that there are no  missing values in the given dataset which is pretty much impressive as it indicates a clean data


# In[ ]:


forest_new.rename(columns = {'Wilderness_Area1':'Rawah', 'Wilderness_Area2':'Neota','Wilderness_Area3':'Comanche_Peak ','Wilderness_Area4':'Cache_la_Poudre'}, inplace = True) 


# In[ ]:


forest_new.columns


# In[ ]:


#Combining the four wilderness area columns and fourty soil type columns to Wild_area and Soil_type respectively, and removing already existing ones.
forest_new['Wild_area'] = (forest_new.iloc[:, 11:15] == 1).idxmax(1)
forest_new['Soil_type'] = (forest_new.iloc[:, 15:55] == 1).idxmax(1)
forestnew=forest_new.drop(columns=['Rawah','Neota',
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


forestnew.columns


# In[ ]:


forestnew.describe()


# # Data Analysis

#  As per the initial analysis.
#  
# ##The average elevtion is around 2750m with value ranging from 1863m to 3849m.
# ##The mean horizontal distance to surface water features is 227 units and for vertical distance 51 units.
# ##Mean horizontal distance to roadways is 1714 units. The values range from 0 units to 6890 units and hence the standard deviation is 1325 units.
# ##Mean horizontal distance to firepoints is 1511 units. The values range from 0 to 6993 units.

# In[ ]:


#Count of number of entries of each cover type.
sns.countplot(forestnew['Cover_Type'],color="red");

# count plot displayed above shows that all forest types are almost equally distributed. 
# In[ ]:


#Count of the entries from different wilderness areas.
sns.countplot(forestnew['Wild_area'], palette= 'spring');

# Data entries from Comanche Peak wild area is more frequent while the entries from Neota wild area is very small. It is also clear that the entries from all 4 wild area have different frequency of occurence. Although the samples from each cover type is same, the sampling within each wilderness area must have been irregular.
# In[ ]:


#Distribution of elevation values in the data.
sns.distplot(forestnew['Elevation'],kde=False,color='green', bins=100);
plt.ylabel('Frequency',fontsize=10)


# The elevation values range between 2000m to 4000m. Thus the data is collected from relatively high altitude areas, as indicated by the vegetation type observed in the data. The distribution of elevation values follows a trimodal fashion, peaking in the middle of the intervals 2000m-2500m, 2500m-3000m,3000m-3500m, and tapering at both ends.

# In[ ]:


#Distribution of aspect values in the data
sns.distplot(forestnew['Aspect'],kde=False,color='green', bins=100);
plt.ylabel('Frequency',fontsize=10)


# In[ ]:


#Distribution of frequency of various soil types in the data.
forestnew['Soil_type'].value_counts().plot(kind='barh',figsize=(10,10));
plt.xlabel('Frequency',fontsize=10)
plt.ylabel('Soil_types',fontsize=10)

##Of the forty soil types mentioned, two, namely soil-type 8 and soil-type 25 are not present in the data.
##Soil type 28 is present in the smallest amount.
##Soil-type 10 is present in the maximum samples. 
##This indicates the wide differences in the representaion of various soil types in the region of interest.
# In[ ]:


#Boxplot between elevation and Cover type
sns.boxplot(y=forest_new['Elevation'],x=forest_new['Cover_Type'],palette='rainbow')

# Here according to this box plot forest cover type 7(Krummholz)and 1(Spruce/Fir) has higher elevation than others
 while type 4(Cottonwood/Willow) has the lowest among all.

# In[ ]:


#Boxplot between Aspect and Cover type
sns.boxplot(y=forest_new['Aspect'],x=forest_new['Cover_Type'],palette='spring')

# while comparing the aspect ratios of different forest cover type ,its obseved that the cover type 6 has the highest among       all. 
# Most types have their aspect ratios around 100 and 200.
# In[ ]:


#Boxplot between Slope and Cover type
sns.boxplot(y=forest_new['Slope'],x=forest_new['Cover_Type'],palette='spring')

## slope do not vary significantly among the forest cover types.
## almost all covertypes have a mean slope between 12 and 25 with pretty good number of outliers in case of types 1,2,5,and 7.
# In[ ]:


#Boxplot between Horizontal_Distance_To_Hydrology and Cover type
sns.boxplot(y=forest_new['Horizontal_Distance_To_Hydrology'],x=forest_new['Cover_Type'],palette='rainbow')

#The median of horizontal distances to surface water features vary from as low as 0 to nearly 200.
##4(Cottonwood/Willow) is present close to water source based on this data.
##7(Krummholz) is present most distant to water sources.
# In[ ]:


#Boxplot between Vertical_Distance_To_Hydrology and Cover type
sns.boxplot(y=forest_new['Vertical_Distance_To_Hydrology'],x=forest_new['Cover_Type'],palette='rainbow')

# As its clear from the above plot there is no much variation in the vertical distance to hydrology among the forest types.
#Therefore further analysis on vertical distance to water source may be avoided.
# In[ ]:


#Boxplot between Horizontal_Distance_To_Roadways and Cover type
sns.boxplot(y=forest_new['Horizontal_Distance_To_Roadways'],x=forest_new['Cover_Type'],palette='spring')

# Most forest types is around an average distance of 2000 units from the roadways except for 7.

##This shows that on an average, most types are equidistant from roadways,which implies that they are prone to human interference and explotation.
 




# In[ ]:


#Boxplot between Hillshade_9am and Cover type
sns.boxplot(y=forest_new['Hillshade_9am'],x=forest_new['Cover_Type'],palette='rainbow')

# Hillshade at 9am values across forest types lie between 200 and 250
# In[ ]:


#Boxplot between Hillshade_Noon and Cover type
sns.boxplot(y=forest_new['Hillshade_Noon'],x=forest_new['Cover_Type'],palette='rainbow')


# In[ ]:


##Hillshade at noon values across forest types are similar, lying between 220 to 240.


# In[ ]:


#Boxplot between Hillshade_3pm and Cover type
sns.boxplot(y=forest_new['Hillshade_3pm'],x=forest_new['Cover_Type'],palette='rainbow')

# Hillshade at 3pm values across forest types vary between 100 to 150.
# In[ ]:


#Boxplot between Horizontal_Distance_To_Fire_Points and Cover type
sns.boxplot(y=forest_new['Horizontal_Distance_To_Fire_Points'],x=forest_new['Cover_Type'],palette='rainbow')


# In[ ]:


##The median horizontal distance to firepoints varies from 1000-2000 units across all the foresttypes except for type 7.
## close proximity to firepoints could be due to the influence of roadways and thus human interference.


# In[ ]:


#Creating data frame for Degree Variables 
X_deg=forest_new[['Elevation','Aspect','Slope','Cover_Type']]


# In[ ]:


X_deg


# In[ ]:


#Creating pairplot for Degree Variables
sns.pairplot(X_deg,hue='Cover_Type',palette='ocean')


# In[ ]:


X_dist=forest_new[['Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points','Cover_Type']]


# In[ ]:


#Creating pairplot for Degree Variables
sns.pairplot(X_dist,hue='Cover_Type',palette='spring')


# In[ ]:


#Creating data frame for Hillshade Variables 
X_hs=forest_new[['Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Cover_Type']]


# In[ ]:


#Creating data frame for Hillshade Variables 
X_wild=forest_new[['Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4','Cover_Type']]


# In[ ]:


#Creating pairplot for Hillshade Variables
sns.pairplot(X_wild,hue='Cover_Type',palette='spring')


# In[ ]:


#Elevation values across wilderness areas.
sns.boxplot(y=forestnew['Elevation'],x=forestnew['Wild_area']);


#there is no much difference in the elevation of  Rawah and Comanche peak areas. The difference in the elevation of these regions are not very distinct.
##Cache la Poudre area has the lowest elevation and is quite distinct from others.
##This might influence the vegetation type in these regions. 
# In[ ]:


#Elevation values across wilderness areas and forest cover types.
sns.catplot(data=forestnew,x='Wild_area',y="Elevation",hue="Cover_Type");


# ##This plot shows that the forest cover type 7(Krummholz) which is a high elevataion forest cover,is absent in the low elevation Cache la Poudre area.
# ##4(Cottonwood/willow) tree is present only in this low elevation region.
# ##It is clear that the point of highest elevation point is present in Comanche Peak region eventhough Neota area has the highest median elevation.
# ##Krummholz forest type is the only forest cover type present from 3500m to 3750m elevation. 

# In[ ]:


#Horizontal distance to fire points across wilderness areas.
sns.catplot(data=forestnew,x='Wild_area',y="Horizontal_Distance_To_Fire_Points");

##Cache la Poudre area is closest to firepoints as seen from the plot.This means that the vegetation here is more prone to fire.
# In[ ]:


#Horizontal distance to fire points across wilderness areas and forest cover types.
sns.catplot(data=forestnew,x='Wild_area',y="Horizontal_Distance_To_Fire_Points",hue="Cover_Type");

##Comanche peak area 7(Krummholz) is farthest from firepoint while in Neota area, this forest cover type is closer to firepoints.
##Vegetation in Cache la Poudre area are relatively closer to fire points.
# In[ ]:


##Horizontal distance to roadways across wilderness areas.
sns.boxplot(y=forestnew['Horizontal_Distance_To_Roadways'],x=forestnew['Wild_area']);

##Neota area, eventhough is at a higher median elevation than other areas, is closer to roadways.
# In[ ]:


#Elevation values across horizontal distance to fire points.
sns.lmplot(data= forestnew,x='Elevation',y='Horizontal_Distance_To_Fire_Points',scatter=False,palette='magma');


# In[ ]:


#Elevation values across horizontal distance to roadways and elevation.
sns.lmplot(data=forestnew,x='Elevation',y='Horizontal_Distance_To_Roadways',scatter=False);


# In[ ]:


#Relationship between horizontal distance to roadways and firepoint.
sns.lmplot(data=forestnew,x='Horizontal_Distance_To_Fire_Points',y='Horizontal_Distance_To_Roadways',scatter=False);

#the above three plots shows that with elevation, distance to firepoints and roadways increases. 
##It also shows that, on average, horizontal distance to fire points and distance to roadways are directly proportional.
# In[ ]:


#Relationship between distance to firepoints and roadways across wilderness types.
sns.lmplot(data=forestnew,x='Horizontal_Distance_To_Fire_Points',y='Horizontal_Distance_To_Roadways',scatter=False,hue="Wild_area");

##From this plot, we can understand that the relationship between distance to firepoints and distance to roadways are directly proportional, except in Cache la Poudre area.
# In[ ]:


sns.lmplot(x='Horizontal_Distance_To_Fire_Points',y='Horizontal_Distance_To_Roadways',hue="Cover_Type",scatter=False,data= forestnew, palette= 'winter');

# Horizontal distance to fire points and horizontal distance are directly proportional.
#This might be due to human influence on fire point.

# In[ ]:


sns.catplot(y= 'Soil_type', hue= 'Cover_Type', kind= 'count', palette='spring', height= 15, data=forestnew) 

the graph above shows the soil types in which the seven forest cover types are present.
==> 1(Spruce/Fir) is present in the highest frequency in soil type 29(Como - Legault families complex, extremely stony).
==> 2(Lodgepole Pine) is present in the highest frequency in soil type 29(Como - Legault families complex, extremely stony).
==> 3(Ponderosa Pine) is present in the highest frequency in soil type 10(Bullwark - Catamount families - Rock outcrop complex, rubbly).
==> 4(Cottonwood/Willow) is present in the highest frequency in soil type 3( Haploborolis - Rock outcrop complex, rubbly).
==> 5(Aspen) is present in the highest frequency in soil type 30(Como family - Rock land - Legault family complex, extremely stony).
==> 6(Douglas-fir) is present in the highest frequency in soil type 10(Bullwark - Catamount families - Rock outcrop complex, rubbly).
==> 7(Krummholz) is present in the highest frequency in soil type 38(Leighcan - Moran families - Cryaquolls complex, extremely stony).
# In[ ]:


sns.catplot(y="Soil_type",hue="Wild_area",kind="count",palette="winter",height=10,data= forestnew);


# In[ ]:





1 - Spruce/Fir.. Present in all wilderness areas except Cache la Pourde, it is present in the highest frequency in the Rawah wilderness area. It occurs at a relatively high elevation of around 3125m, second only to Krummholz forest type. It is present in the highest frequency in soil type 29(Como - Legault families complex, extremely stony), but is present in other soil types as well. It is present relatively farther from fire points and roadways. This could mean that this is a high altitude tree that has less human interference. It is not very far from surface water sources, but not very close either.2 - Lodgepole Pine.. Present in all wilderness although predominantly present in Rawah. Not very close to water sources or firepoints. Close to roadways in Cache la Poudre and Neota area.Elevation range between 2500-3250. Present in highest frequency in soil type 29(Como - Legault families complex, extremely stony).3 - Ponderosa Pine..Present only in Comanche and Cache la Paudre. Elevation value highest at 2750m. Present in highest frequency in soil type 10(Bullwark - Catamount families - Rock outcrop complex, rubbly). Close to water sources. Also close to roadways and firepoints, therefore prone to human interfernces and forest fire.4-Cottonwood/Willow.. Present only in Cache la Poudre. Elevation value highest at around 2350m. Present in highest frequency in soil type 3( Haploborolis - Rock outcrop complex, rubbly). Close to water sources. Also close to roadways and firepoints, therefore prone to human interfernces and forest fire.5 - Aspen..Present in Rawah and Comanche, although predominantly in Rawah. Elevation value highest at around 2900m. Present in high frequency in soil type 30(Como family - Rock land - Legault family complex, extremely stony). Close to water bodies, raodways and firepoints.

6 - Douglas-fir..Present only in Comanche peak and Cache la Poudre. Elevation value highest at around 2900m. Present in high frequency in soil type 10(Bullwark - Catamount families - Rock outcrop complex, rubbly). Close to water bodies, raodways and firepoint7 - Krummholz..Present in all wild areas except Cache la Paudre.It has the highest elevation value above 3750m. Present in high frequency in soil type 38(Leighcan - Moran families - Cryaquolls complex, extremely stony). It is far from roadways except in area Neota. It is most distant to water sources. It is closer to firepoint in the case of Neota and further away in case Rawah and Comanche peaks
# In[ ]:




