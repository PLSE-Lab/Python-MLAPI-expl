#!/usr/bin/env python
# coding: utf-8

# #      Exploratory Data Analysis (EDA) of Forest Cover Type Data
The study area includes four wilderness areas located in the Roosevelt National Forest of Northern Colorado.

The wilderness areas are:

1 - Rawah Wilderness Area
2 - Neota Wilderness Area
3 - Comanche Peak Wilderness Area
4 - Cache la Poudre Wilderness Area.


The seven forest cover types are:

1 - Spruce/Fir
2 - Lodgepole Pine
3 - Ponderosa Pine
4 - Cottonwood/Willow
5 - Aspen
6 - Douglas-fir
7 - Krummholz.
# # Preparing the data for analysis

# In[ ]:


#Importing the required libraries.
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


#Reading the given csv file into a dataframe.
df=pd.read_csv("/kaggle/input/forest_train.csv")


# In[ ]:


#Getting the initial five values of the dataframe.
df.head()


# In[ ]:


#Getting the dimensionality of the dataframe.
df.shape


# In[ ]:


#Getting the summary of the data types in the data frame
df.info()


# In[ ]:


#Renaming the wilderness area columns for better clarity.
df.rename(columns = {'Wilderness_Area1':'Rawah', 'Wilderness_Area2':'Neota','Wilderness_Area3':'Comanche_Peak ','Wilderness_Area4':'Cache_la_Poudre'}, inplace = True) 


# In[ ]:


#Checking the column names
df.columns


# In[ ]:


#Combining the four wilderness area columns and fourty soil type columns to Wild_area and Soil_type respectively, and removing already existing ones.
df['Wild_area'] = (df.iloc[:, 11:15] == 1).idxmax(1)
df['Soil_type'] = (df.iloc[:, 15:55] == 1).idxmax(1)
df_forest=df.drop(columns=['Id','Rawah', 'Neota',
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
df_forest


# In[ ]:


#Checking the columns in the modified dataframe.
df_forest.columns


# 
# # Preliminary analysis

# In[ ]:


#Provide the general descriptive statistical values.
df_forest.describe()


# * The average elevation is around 2750m with values ranging from 1863m to 3849m.
# 
# * The mean horizontal distance to surface water features is 227 units and mean vertical distance to surface water features is 51 units.
# 
# * Mean horizontal distance to roadways is 1714 units. The values range from 0 units to 6890 units and hence the standard deviation is 1325 units.
# 
# * Mean horizontal distance to firepoints is 1511 units. The values range from 0 to 6993 units.

# In[ ]:


#Count of number of entries of each cover type.
sns.countplot(df_forest['Cover_Type'],color="grey");


# All seven forest cover types occur with the same frequency in the data.

# In[ ]:


#Count of the entries from different wilderness areas.
sns.countplot(df_forest['Wild_area']);


# * The four wilderness areas, on the other hand, occur in varying frequencies in the data.
# * Entries from Comanche Peak wilderness area occur the most and entries from the Neota wilderness area, the least.
# * This points to the uneven sampling with respect to the wilderness areas, even though the number of samples for each cover type is equal.

# In[ ]:


#Distribution of elevation values in the data.
sns.distplot(df_forest['Elevation'],kde=False,color='red', bins=100);
plt.ylabel('Frequency',fontsize=10)


# * The elevation values range between 2000m to 4000m. Thus the data is collected from relatively high altitude areas, as indicated by the vegetation type observed in the data.
# * The distribution of elevation values follows a trimodal fashion, peaking in the middle of the intervals 2000m-2500m, 2500m-3000m,3000m-3500m, and tapering at both ends.

# In[ ]:


#Distribution of aspect values in the data
sns.distplot(df_forest['Aspect'],kde=False,color='red', bins=100);
plt.ylabel('Frequency',fontsize=10)


# The aspect values range from 0 to 350 with most of them lying between 0-100 and 275-350.
# It follows a bimodal distribution.

# In[ ]:


#Distribution of values of slope in the data.
sns.distplot(df_forest['Slope'],kde=False,color='red', bins=100);
plt.ylabel('Frequency',fontsize=10)


# A positively skewed (right skewed) distribution, peaked at around 10, is obtained with the slope values in the data.

# In[ ]:


#Distribution of values of the horizontal distance to roadways.
sns.distplot(df_forest['Horizontal_Distance_To_Roadways'],kde=False,color='blue', bins=100);
plt.ylabel('Frequency',fontsize=10)


# * For horizontal distance from roadways, a positively skewed distribution is obtained, peaked at 1000. 
# * It is thus clear from the graph that most of the samples are within 0-2000 distance to the roadways.
# * This indicates a considerable human impact and chances of commercial exploitation of the forests.

# In[ ]:


#Distribution of values of the horizontal distance to fire points.
sns.distplot(df_forest['Horizontal_Distance_To_Fire_Points'],kde=False,color='blue', bins=100);
plt.ylabel('Frequency',fontsize=10)


# * Most of the samples are located within a distance of 0-2000 units to fire points in this positively skewed distribution.This indicates the horizontal distance to wildfire ignition point.
# * This data, thus indicates the noticeable influence of human activities including presence of roads and proximity to fire points.The type of vegetaion that would grow in these regions would depend on these factors.

# In[ ]:


#Distribution of values of the horizontal distance to nearest surface water features.
sns.distplot(df_forest['Horizontal_Distance_To_Hydrology'],kde=False,color='green', bins=100);
plt.ylabel('Frequency',fontsize=10)


# * The horizontal distances to the nearest surface water features also shows a positively skewed distribution,ranging from 0 to 1200, peaking near zero. This means that most of the samples are present very close to surface water sources.

# In[ ]:


#Distribution of values of the vertical distances to nearest surface water features.
sns.distplot(df_forest['Vertical_Distance_To_Hydrology'],kde=False,color='green', bins=100);
plt.ylabel('Frequency',fontsize=10)


# * The vertical distance to surface water features,even though a positively skewed distribution with a sharp peak at zero,ranges from -150 to around 500.

# In[ ]:


#Distribution of values of the hillshade index at 9am during summer solstice on an index from 0-255.
sns.distplot(df_forest['Hillshade_9am'],kde=False,color='black', bins=100);
plt.ylabel('Frequency',fontsize=10)


# * The values for hillshade index at 9am follows a negtively skewed distribution, peaking at around 225, ranging from 100 to 250.

# In[ ]:


#Distribution of values of the hillshade index at noon during summer solstice on an index from 0-255.
sns.distplot(df_forest['Hillshade_Noon'],kde=False,color='black', bins=100);
plt.ylabel('Frequency',fontsize=10)


# * The values for hillshade index at noon follows a negtively skewed distribution, peaking at around 225, ranging from 125 to 250.

# In[ ]:


#Distribution of values of the hillshade index at 3pm during summer solstice on an index from 0-255.
sns.distplot(df_forest['Hillshade_3pm'],kde=False,color='black', bins=100);
plt.ylabel('Frequency',fontsize=10)


# * The values for hillshade index at 3pm follows a more or less symmetric normal distribution, peaking at around 150,ranging from 0 to 250.
# 

# In[ ]:


#Distribution of frequency of various soil types in the data.
df_forest['Soil_type'].value_counts().plot(kind='barh',figsize=(10,10));
plt.xlabel('Frequency',fontsize=10)
plt.ylabel('Soil_types',fontsize=10)


# * Of the forty soil types mentioned, two, namely soil-type 8 and soil-type 25 are not present in the data.
# 
# * Soil type 28 is present in the smallest amount.
# 
# * Soil-type 10 is present in the maximum samples. 
# 
# * This indicates the wide differences in the representaion of various soil types in the region of interest.

# **The above analysis of the distribution of various features gives a clear idea of the geography of the area of interest.The key take away at this point are as follows:**

# * This is a relatively high elavation area with a mean elavation around 2750m.
# 
# * Located close to roadways and fire points,there are high chances of human intereference and significant impact like forest fires in this region.
# 
# * Surface water features are present close to the sample areas.
# 
# * Bullwark - Catamount families - Rock outcrop complex, rubbly type of soil is the most common soil type in the data.

# **Now let us explore the relationships between various forest cover types to other features to better understand their variation and relevance to or analysis.**

# # Detailed analysis

# **Analysis of features across various forest cover types.**

# In[ ]:


#Forest cover types in each wilderness areas.
a1=sns.countplot(data=df_forest,x='Wild_area',hue="Cover_Type");
a1.set_xticklabels(a1.get_xticklabels(),rotation=15);


# * Rawah wilderness area has only the forest cover types 1(Spruce/Fir),2(Lodgepole Pine),5(Aspen) and 7(Krummholz).
# 
# * Comanche peak wilderness area has all the forest cover types, except 4.Neota wilderness area has only 3 forest cover types-1(Spruce/Fir),2(Lodgepole Pine) and 7(Krummholz).So, Neota and Comanche Peak are two extremes in forest cover type diversity, having the lowest and highest repsectively.
# 
# * The forest cover type 4(Cottonwood/Willow) is present only in the Cache la Poudre wilderness area and is the major forest cover in that area.It is the rarest in terms of distribution, but has the highest count than any other cover type in a single wilderness region.ache la Poudre wilderness is devoid of 3 forest cover types(Spruce/Fir,Aspen,Krummholz).
# 
# * Forest cover type 2(Lodgepole Pine) is present in all wilderness regions, maximum in Rawah and minimum in Cache la Poudre wilderness.

# In[ ]:


#Elevation values across forest types.
sns.boxplot(y=df_forest['Elevation'],x=df_forest['Cover_Type']);


# * The box plot of cover type versus elevation reveals the clear dispersion of various forest cover types based on elavation.Most of the tpes co exist at similar elavations, except for forest cover type 7(Krummholz).
# 
# * The forest cover type 7(Krummholz) is present at the highest median elavation of around 3375m. It is followed by forest cover type 1(Spruce/Fir) at nearly 3125m median elavation.
# 
# * Forest cover type 2(Lodgepole Pine) and 5(Aspen) occur at similar elavations. Similarly,6(Douglas-fir) and 3 (Ponderosa Pine) occur at the same elavations.
# 
# * The forest cover type 4 (Cottonwood/Willow) shows the least median elavation at around 2250m.

# In[ ]:


#Aspect values across forest types.
sns.boxplot(y=df_forest['Aspect'],x=df_forest['Cover_Type']);


# * The aspect values do not vary very distinctly with forest cover types.The median aspect values for all the types occur between 100-200.
# 
# * This parameter might not be very significant to come up with any conclusions regarding the area of interest.

# In[ ]:


#Slope values across forest types.
sns.boxplot(y=df_forest['Slope'],x=df_forest['Cover_Type']);


# * The values of slope do not vary very significantly between various forest cover types, as we can observe that the box plots overlap.Slight variations in median slope is observed.

# In[ ]:


#Soil types across forest types.
a2=sns.catplot(y="Soil_type",hue="Cover_Type",kind="count",palette="pastel",height=15,data=df_forest);


# This graph showcasses the soil types in which the seven forest cover types are present.
# 
# * 1(Spruce/Fir) is present in the highest frequency in soil type 29(Como - Legault families complex, extremely stony).
# 
# * 2(Lodgepole Pine) is present in the highest frequency in soil type 29(Como - Legault families complex, extremely stony).
# 
# * 3(Ponderosa Pine) is present in the highest frequency in soil type 10(Bullwark - Catamount families - Rock outcrop complex, rubbly).
# 
# * 4(Cottonwood/Willow) is present in the highest frequency in soil type 3( Haploborolis - Rock outcrop complex, rubbly).
# 
# * 5(Aspen) is present in the highest frequency in soil type 30(Como family - Rock land - Legault family complex, extremely stony).
# 
# * 6(Douglas-fir) is present in the highest frequency in soil type 10(Bullwark - Catamount families - Rock outcrop complex, rubbly).
# 
# * 7(Krummholz) is present in the highest frequency in soil type 38(Leighcan - Moran families - Cryaquolls complex, extremely stony).
# 

# In[ ]:


#Horizontal distance to fire points across forest types.
sns.boxplot(data=df_forest,x='Cover_Type',y="Horizontal_Distance_To_Fire_Points");


# * The median horizontal distance to firepoints varies from 1000-2000 units across the various forest cover types.
# 
# * It is the least for type 3,4 and 6.
# 
# * This shows that on average, all types are vulnerable to the occurance of forest fire.

# In[ ]:


#Horizontal distance to roadways across forest types.
sns.boxplot(y=df_forest['Horizontal_Distance_To_Roadways'],x=df_forest['Cover_Type']);


# * The median horizontal distance to roadways varies from 1000-2000 units across all the forest cover types except 7.
# 
# * This shows that on average, most types are equidistant from roadways. This would imply that they are prone to human interference and explotation.
# 
# * Since the distance to firepoints are also in this range, one could assume that both distance to roadways and firepoints influence each other. 

# In[ ]:


#Horizontal distance to surface water features across forest types.
sns.boxplot(y=df_forest['Horizontal_Distance_To_Hydrology'],x=df_forest['Cover_Type']);


# * The median of horizontal distances to surface water features vary from as low as 0 to nearly 200.
# 
# * 4(Cottonwood/Willow) is present close to water source based on this data.
# 
# * 7(Krummholz) is present most distant to water sources.

# In[ ]:


#Vertical distance to surface water features across forest types.
sns.boxplot(y=df_forest['Vertical_Distance_To_Hydrology'],x=df_forest['Cover_Type']);


# * The vertical distance to water sources do not vary much among various forest cover types.
# 
# * This might mean that the vertical distance to water source is not an important factor in performing further analysis on our data.

# In[ ]:


#Hillshade at 9am values across forest types.
sns.boxplot(y=df_forest['Hillshade_9am'],x=df_forest['Cover_Type']);


# * Hillshade at 9am values across forest types lie between 200 and 250.

# In[ ]:


#Hillshade at noon values across forest types.
sns.boxplot(y=df_forest['Hillshade_Noon'],x=df_forest['Cover_Type']);


# * Hillshade at noon values across forest types are similar, lying between 220 to 240.

# In[ ]:


#Hillshade at 3pm values across forest types.
sns.boxplot(y=df_forest['Hillshade_3pm'],x=df_forest['Cover_Type']);


# * Hillshade at 3pm values across forest types vary between 100 to 150.

# * This shows that hillshade values at 9 am and noon are in similar range of around 200 to 250, while hillshade values at 3pm are lower,in the range of 100 to 150 with rspect to various forest cover types.

# **Now we will explore the variation of various features and then combine it with forest cover types to get a wholesome final idea on the area of interest.**

# In[ ]:


#Elevation values across wilderness areas.
a2=sns.boxplot(y=df_forest['Elevation'],x=df_forest['Wild_area']);
a2.set_xticklabels(a2.get_xticklabels(),rotation=15);


# * The area with the highest elevation is Neota wilderness area.
# 
# * This is followed by Rawah and Comanche peak areas. The difference in the elevation of these regions are not very distinct.
# 
# * Cache la Poudre area has the lowest elevation and is quite distinct from others.
# 
# * This might influence the vegetation type in these regions. 

# In[ ]:


#Elevation values across wilderness areas and forest cover types.
a3=sns.catplot(data=df_forest,x='Wild_area',y="Elevation",hue="Cover_Type");
a3.set_xticklabels(rotation=65, horizontalalignment='right');


# * This plot shows that the forest cover type 7(Krummholz) which is a high elevataion forest cover,is absent in the low elevation Cache la Poudre area.
# 
# * 4(Cottonwood/willow) tree is present only in this low elevation region.
# 
# * It is clear that the point of highest elevation is present in Comanche Peak region eventhough Neota area has the highest median elevation.
# 
# * Krummholz forest type is the only forest cover type present from 3500m to 3750m elevation. 

# In[ ]:


#Horizontal distance to fire points across wilderness areas.
a4=sns.boxplot(data=df_forest,x='Wild_area',y="Horizontal_Distance_To_Fire_Points");
a4.set_xticklabels(a4.get_xticklabels(),rotation=15);


# * Cache la Poudre area is closest to firepoints as seen from the plot.This means that the vegetation here is more prone to fire.
# 
# * Rawah area is the farthest from firepoints.

# In[ ]:


#Horizontal distance to fire points across wilderness areas and forest cover types.
a5=sns.catplot(data=df_forest,x='Wild_area',y="Horizontal_Distance_To_Fire_Points",hue="Cover_Type");
a5.set_xticklabels(rotation=65, horizontalalignment='right');


# * From this plot we can understand that in Rawah area, cover type 2(Lodgepole Pine) is present in large frequency farthest from firepoints.
# 
# * In Comanche peak area 7(Krummholz) is farthest from firepoint while in Neota area, this forest cover type is closer to firepoints.
# 
# * Vegetation in Cache la Poudre area are relatively closer to fire points.

# In[ ]:


##Horizontal distance to roadways across wilderness areas.
a6=sns.boxplot(y=df_forest['Horizontal_Distance_To_Roadways'],x=df_forest['Wild_area']);
a6.set_xticklabels(a6.get_xticklabels(),rotation=15);


# * This plot shows that Rawah area is the farthest from roadways and Cache la Poudre area is the closest to roadways.
# 
# * Neota area, even though is at a higher median elevation than other areas, is closer to roadways.

# In[ ]:


#Relationship between wild areas and distance to roadways across forest cover types.
a7=sns.catplot(data=df_forest,x='Wild_area',y="Horizontal_Distance_To_Roadways",hue="Cover_Type");
a7.set_xticklabels(rotation=65, horizontalalignment='right');


# * This graph further clarifies our analysis on the relationship of these features.
# 
# * 7(Krummholz) forest cover type is in general far from roadways.
# 
# `Roadways are important areas of human interaction and movement. So, it would be logical to see the relationship between distance to roadways and firepoint.`

# In[ ]:


#Elevation values across horizontal distance to fire points.
sns.lmplot(data= df_forest,x='Elevation',y='Horizontal_Distance_To_Fire_Points',scatter=False);


# In[ ]:


#Elevation values across horizontal distance to roadways and elevation.
sns.lmplot(data=df_forest,x='Elevation',y='Horizontal_Distance_To_Roadways',scatter=False);


# In[ ]:


#Relationship between horizontal distance to roadways and firepoint.
sns.lmplot(data=df_forest,x='Horizontal_Distance_To_Fire_Points',y='Horizontal_Distance_To_Roadways',scatter=False);


# * The above three plots shows that with elevation, distance to firepoints and roadways on average, increases. 
# 
# * It also shows that, on average, horizontal distance to fire points and distance to roadways are directly proportional.
# 
# * This means that these features, namely, Elevation, horizontal distance to roadways and horizontal distance to firepoints are closely related to each other and have a positive correlation.
# 
# `Now, let us analyse this further so as to a better picture.`

# In[ ]:


#Relationship between distance to firepoints and roadways across wilderness types.
sns.lmplot(data=df_forest,x='Horizontal_Distance_To_Fire_Points',y='Horizontal_Distance_To_Roadways',scatter=False,hue="Wild_area");


# From this plot, we can understand that the relationship between distance to firepoints and distance to roadways are directly proportional, except in Cache la Poudre area.

# In[ ]:


#Relationship between distance to firepoints and roadways across forest cover types.
sns.lmplot(data=df_forest,x='Horizontal_Distance_To_Fire_Points',y='Horizontal_Distance_To_Roadways',scatter=False,hue="Cover_Type");


# This graph explains the exception of Cache la Poudre area in the previous graph. 
# 
# * This is beacuse 4(Cottonwood/Willow) is present only in this region and it shows neagtive correlation between firepoint and roadways distances.
# * This must be viewed in the background that Cache la Poudre area is closest to roadways and firepoints.

# * On average, we can say that horizontal distance to fire points and horizontal distance are directly proportional.This might be due to human influence on fire point.
# 
# * 4(Cottonwood/Willow) forest cover type and consequently Cache la Poudre area does not follow the normal trend even though this area is closest to roadways and firepoints.
# 
# * This might be because this forest type might be innately highly inflammable and does not require human interference for starting a fire.

# In[ ]:


##Horizontal distance to hydrology across wilderness areas.
a8=sns.boxplot(y=df_forest['Horizontal_Distance_To_Hydrology'],x=df_forest['Wild_area']);
a8.set_xticklabels(a8.get_xticklabels(),rotation=15);


# * All the forest areas considered are more or less equally distanced from water source.
# * Neota area is slightly far fom surface water source on average.

# In[ ]:


#Relationship between wild areas and distance to hydrology across forest cover types.
a9=sns.catplot(data=df_forest,x='Wild_area',y="Horizontal_Distance_To_Hydrology",hue="Cover_Type");
a9.set_xticklabels(rotation=65, horizontalalignment='right');


# * We observe that the forest cover type 7(Kremmholz) is more spread out from 0 to 1000 distance units in each of the wilderness areas(except in Cache la Poudre).

# In[ ]:


#Elevation values across horizontal distance to fire points.
sns.lmplot(data= df_forest,x='Elevation',y='Horizontal_Distance_To_Hydrology',scatter=False);


# * The above plots shows that elevation and horizontal distance to hydrology are directly proportional. 
# 
# * This, coupled with the previous plots shows that elevation, horizontal distances to roadways, surface water features and firepoints are directly proprtional on average and has a strong influence on the forest cover type and distribution in four different wilderness areas.
Now let us explore the features which were seen not to change much with forest cover type.
# In[ ]:


##Vertical distance to hydrology across wilderness areas.
a10=sns.boxplot(y=df_forest['Vertical_Distance_To_Hydrology'],x=df_forest['Wild_area']);
a10.set_xticklabels(a10.get_xticklabels(),rotation=15);


# * All the wilderness areas show similar vertical distance to surface water features at a mean of around 50 units.

# In[ ]:


#Relationship between wild areas and vertical distance to hydrology across forest cover types.
a11=sns.catplot(data=df_forest,x='Wild_area',y="Vertical_Distance_To_Hydrology",hue="Cover_Type");
a11.set_xticklabels(rotation=65, horizontalalignment='right');


# * As in the previous graph, the difference between various wilderness areas is not very distinct.

# In[ ]:


##Hillshade at 9am across wilderness areas.
a12=sns.boxplot(y=df_forest['Hillshade_9am'],x=df_forest['Wild_area']);
a12.set_xticklabels(a10.get_xticklabels(),rotation=15);


# ##This shows that the median values of hillshade at 9am occurs in the range 200-250 with very less variation among various wilderness areas.

# In[ ]:


##Hillshade at noon across wilderness areas.
a13=sns.boxplot(y=df_forest['Hillshade_Noon'],x=df_forest['Wild_area']);
a13.set_xticklabels(a10.get_xticklabels(),rotation=15);


# * This shows that the median values of hillshade at noon occurs in the range 200-240 with very less variation among various wilderness areas.

# In[ ]:


##Hillshade at 3pm across wilderness areas.
a14=sns.boxplot(y=df_forest['Hillshade_3pm'],x=df_forest['Wild_area']);
a14.set_xticklabels(a10.get_xticklabels(),rotation=15);


# * This shows that the median values of hillshade at 3pm occurs in the range 100-150 with very less variation among various wilderness areas.

# In[ ]:


##Slope values across wilderness areas.
a15=sns.boxplot(y=df_forest['Slope'],x=df_forest['Wild_area']);
a15.set_xticklabels(a10.get_xticklabels(),rotation=15);


# * This shows that the median values of slope occurs in the range 10-20 degrees with very less variation among various wilderness areas.

# In[ ]:


##Aspect values across wilderness areas.
a16=sns.boxplot(y=df_forest['Aspect'],x=df_forest['Wild_area']);
a16.set_xticklabels(a10.get_xticklabels(),rotation=15);


# This shows that the median values of aspect occurs in the range 50-150 degrees.

# * Since the above features: hillshade at 9am,hillshade at noon,hillshade at 3pm,slope and aspect are found to not vary with forest cover type and wilderness area,further analysis of these features might not be required.

# # Conclusion

# **We can infer the following conclusions about various forest cover types:**

# `1-Spruce/Fir`

# * *Present in all wilderness areas except Cache la Pourde, it is present in the highest frequency in the Rawah wilderness area.*
# 
# * *It occurs at a relatively high elevation of around 3125m, second only to Krummholz forest type.*
# 
# * *It is present in the highest frequency in soil type 29(Como - Legault families complex, extremely stony), but is present in other soil types as well.*
# 
# * *It is present relatively farther from fire points and roadways. This could mean that this is a high altitude tree that has less human interference.*
# 
# * *It is not very far from surface water sources, but not as close to water as Cottonwood/Willow.*

# `2-Lodgepole Pine`

# * *This is the only forest cover type that is present in all the wilderness areas, present in maximum frequency in Rawah wilderness area and minimum in Cache la Poudre.*
# 
# * *It occurs at a relatively medium median elevation of 2875m.*
# 
# * *It is present in the highest frequency in soil type 29(Como - Legault families complex, extremely stony).*
# 
# * *It is present relatively farther from fire points and roadways like Spruce/Fir.*
# 
# * *It is present at similar distance to surface water features as Spruce/Fir.*

# `3-Ponderosa Pine`

# * *This forest cover type is present only in Cache la Poudre and Comanche Peak areas, with highest frequency in Cache la Poudre area.*
# 
# * *It is a relatively low elevation forest type.*
# 
# * *It is present in the highest frequency in soil type 10(Bullwark - Catamount families - Rock outcrop complex, rubbly).*
# 
# * *It is present close to firepoints and roadways. It along with Cottonwood/Willow is present in maximum frequency is present in Cache la Poudre.*
# 
# * *The horizontal distance to surface water features is low, indicating the existence of this vegetation close to rivers, lakes or other surface water features.*

# `4-Cottonwood/Willow`

# * *This forest cover type is present only in Cache la Poudre.*
# 
# * *It is a relatively low elevation forest type.*
# 
# * *It is present in the highest frequency in soil type 3( Haploborolis - Rock outcrop complex, rubbly).*
# 
# * *It is present close to firepoints and roadways but these values are negatively correlated in this cover type.This is unlike other forest types,where distance to firepoint and distance to roadways are directly proportional.This property could be due to the innate inflammability of Cottonwood tree parts.*
# 
# * *The horizontal distance to surface water features is the lowest, indicating the existence of this vegetation close to rivers, lakes or other surface water features.*

# `5-Aspen`

# * *Comanche Peak area has the highest frequency of this forest cover. It is present in Rawah area, but absent in the other two areas.*
# 
# * *It is a relatively medium elevation forest type.*
# 
# * *It is present in the highest frequency in soil type 30(Como family - Rock land - Legault family complex, extremely stony).*
# 
# * *It is present relatively close to firepoints and roadways.*
# 
# * *The horizontal distance to surface water features is also relatively low.*

# `6-Douglas-fir`

# * *Cache la Poudre area has the highest frequency of this forest cover. It is present in Comanche Peak area.*
# 
# * *It is a relatively low elevation forest type.*
# 
# * *It is present in the highest frequency in soil type 10(Bullwark - Catamount families - Rock outcrop complex, rubbly).*
# 
# * *It is present close to firepoints and roadways.*
# 
# * *The horizontal distance to surface water features is also low.*

# `7-Krummholz`

# * *A relatively more widely distributed forest cover type, it is present in the highest frequency in Comanche Peak area*
# 
# * *It is the forest type present in the highest elevations. The median elevation is 3375m.*
# 
# * *It is present in the highest frequency in soil type 38(Leighcan - Moran families - Cryaquolls complex, extremely stony).*
# 
# * *It is present farthest from firepoints and roadways.*
# 
# * *The horizontal distance to surface water features is also the highest.*

# **This detailed analysis leaves us with a complete picture of the features of various forest types and wilderness areas.**
