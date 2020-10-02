#!/usr/bin/env python
# coding: utf-8

# # Contents
# 1. Introduction
# 2. Loading libraries and files
# 3. have a first look at the dataset
# 4. check for missings and duplicates in the dataset
# 5. datacleaning and making new variables
# 6. examine the target feature Cover_Type
# 7. examine the predictors features: continuous features, binary features wilderness areas and soil types
# 
# # 1. Introduction
# 
# The goal of this competition is to predict the type of trees in a certain area based on  various geographical features. The type of trees is represented by the feature *Cover-Type.* The seven cover types are:
# 
# 1. [Spruce/Fir](http://https://en.wikipedia.org/wiki/Spruce-fir_forests)
# 2. [Lodgepole Pine](http://https://en.wikipedia.org/wiki/Pinus_contorta)
# 3. [Ponderosa Pine](http://https://en.wikipedia.org/wiki/Pinus_ponderosa)
# 4. Cottonwood/Willow
# 5. Aspen
# 6. Douglas-fir
# 7. [Krummholz](http://https://en.wikipedia.org/wiki/Krummholz)
# 
# You can click on some of the names to get more information about the type of forest. In this kernel I perform an [exploratory data analysis (EDA)](https://en.wikipedia.org/wiki/Exploratory_data_analysis). The reason for performing an EDA is to summarize the main characteristics of the data and to get meaningfull insights that can help to understand the data.
# 
# In some cases I hide the code to make this EDA more readible.
# 
# !!Work is still in progress!!

# # 2. Load libraries and datafiles

# In[ ]:


# load libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


# load data
train = pd.read_csv("../input/learn-together/train.csv", index_col = 'Id')
test = pd.read_csv("../input/learn-together/test.csv", index_col = 'Id')

# combine train and test data
train['train_test'] = 'train'
test['train_test']  = 'test'

alldata = pd.concat([train.drop(columns = ['Cover_Type']), test])


# # 3. A first look at the dataset

# In[ ]:


print("The train set has {0} rows and {1} columns.".format(str(train.shape[0]), str(train.shape[1])))
print("The test set has {0} rows and {1} columns.".format(str(test.shape[0]), str(test.shape[1])))


# In[ ]:


# set figure size
plt.figure(figsize=(14,6))

#add title
plt.title("Countplot of train versus test data")

# make countplot
sns.countplot(x = alldata.train_test)


# The test data contains more than 500000 rows (observations) and 54 columns. In the test set, the column Cover_Type, the one we have to predict is missing. As shown in the figure above the test set is much larger than the train set. Usually the test set is smaller than the train set.

# In[ ]:


print("The number of unique values per feature in the train set:")
print(train.nunique())


# In[ ]:


print("The number of unique values per feature in the test set:")
print(test.nunique())


# The first ten features have many different unique values. So there are likely to be continuous features. The remaining features all have two unique values, except for Soil_Type7 and Soil_Type15 in the train set, which have one unique value. These features are binary. Since Soil_Type7 and Soil_Type15 have only one unique value need to be removed when building a model to predict Cover_Type.

# # 4. Check for missings and duplicates

# In[ ]:


# get number of missings per column in train and test set
print(train.isnull().sum().sum())
print(test.isnull().sum().sum())


# There are no missings in the train and test set.

# In[ ]:


print(train.duplicated(subset=None, keep='first').sum())
print(test.duplicated(subset=None, keep='first').sum())


# There are no duplicate rows in the data set.

# # 5. Data cleaning and new variables
# 
# Here I perform data cleaning on the data set. This includes making new variables, and dividing the train set into 7 different sets, one for each cover type. You can click on code if you want to see the code used in this section.

# In[ ]:


# make new variable compass
def Compass(row):
    if row.Aspect < 45:
        return 'north'
    elif row.Aspect < 135:
        return 'east'
    elif row.Aspect < 225:
        return 'south'
    elif row.Aspect < 315:
        return 'west'
    else:
        return 'north'
    
train['Compass'] = train.apply(Compass, axis='columns')


# # 5. Examine the target Cover_Type
# 
# The feature we have to predict is called Cover_Type and resembles the type of forest growing at specific site. This feature is only present in the train set. Here I make a new variable called CoverType, instead of number I use the actual naming of the forest types. This makes the charts I make next easier to read and understand. You can click on the code button to see how I made a new variable.

# In[ ]:


def CoverType(row):
    if row.Cover_Type == 1:
        return 'Spruce/Fir'
    elif row.Cover_Type == 2:
        return 'Lodgepole Pine'
    elif row.Cover_Type == 3:
        return 'Ponderosa Pine'
    elif row.Cover_Type == 4:
        return 'Cottonwood/Willow'
    elif row.Cover_Type == 5:
        return 'Aspen'
    elif row.Cover_Type == 6:
        return 'Douglas-fir'
    else:
        return 'Krummholz'

train['CoverType'] = train.apply(CoverType, axis='columns')


# In[ ]:


# set figure size
plt.figure(figsize=(14,6))

#add title
plt.title("Countplot of Cover Types")

# make countplot
sns.countplot(x = train.CoverType)


# Each cover type appears 2160 times, this means we have a balanced training set. We have to keep in mind that in the real world, there is an imbalance in the number of forests. 

# # 6. Examine the predictors
# 
# Here I start with exploring the predictor variables. The predictor variables can be divided in three groups: 
# 
# The continuous features:
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
# 
# Wilderness_Area (4 binary columns, 0 = absence or 1 = presence) - Wilderness area designation
# 
# Soil_Type (40 binary columns, 0 = absence or 1 = presence) - Soil Type designation
# 
# I start with exploring the continuous features.
# 
# ## 6.1 Continuous features
# 
# Here I examine how the continuous features. The first 10 columns are continuous features. So lets describe them first by getting some descriptives (mean, min, max, etc).

# In[ ]:


myvars = list(train.columns[:10])
print("Describing the train set:")
train[myvars].describe().T


# The first feature Elevation gives information about the altitude in meters. The lowest location is at 1863 meters, the highest location is at 3849 meters. The avarage altitude is 2749 meters. 
# 
# The second feature Aspect gives information about aspect in degrees azimuth. This value ranges between 0 and 360 degrees, and its average is 156.67.
# I had no idea what aspect meant, so I did some googling. I found the following website: https://pro.arcgis.com/en/pro-app/tool-reference/3d-analyst/how-aspect-works.htm that explains how aspect works. According to the information found on the website, aspect identifies the direction of the downhill slope faces. It is measured clockwise in degrees from 0 (due north) to 360 (again due north), coming full circle. At this point I have no idea if this is correct.
# 
# The feature Slope gives the Slope in degrees and is thus a measure how steap the location is. This features ranges from 0 (flat area) to 52 degrees. The avarage slope is 16.5 degrees.
# 
# The feature Horizontal_Distance_To_Hydrology gives information about the horizontal distance to nearest surface water. Its value ranges between 0 and 1343 meters. On avarage this value is 227 meters.
# 
# The feature vertical distance to hydrology gives information about the vertical distance to the nearest surface water. This feature contains negative values. At this point I cannot explain these negative values.
# 
# The feature Horizontal_Distance_To_Roadways gives information about how far a location is from the nearest roadway. It ranges between 0 and 6890. On avarage a location is 1714 meters from the nearest roadway.
# 
# There are three features that give information about [hillshade index](http://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-hillshade-works.htm):
# Hillshade_9am (0 to 255 index) - Hillshade index at 9am, summer solstice
# Hillshade_Noon (0 to 255 index) - Hillshade index at noon, summer solstice
# Hillshade_3pm (0 to 255 index) - Hillshade index at 3pm, summer solstice
# Hillshade_9am ranges from 0 to 254, with an avarage of 213. Hillshade_Noon ranges from 99 to 254 with an avarage of 219. Hillshade_3pm ranges from 0 to 248 with an avarage of 135. Hillshade_index can be used to tell if a site falls in the shadow. Hillshade index goes from dark to bright (0 to 255)!
# 
# Horizontal_Distance_To_Fire_Points represents the horizontal distance to nearest wildfire ignition points. On avarage this is 1511 meters, with a minimum of 0 and a maximum of 6993 meters.
# 
# Relevant website for understanding azimuth and hillshade: https://pro.arcgis.com/en/pro-app/tool-reference/3d-analyst/how-hillshade-works.htm

# ### 6.1.1 Elevation

# In[ ]:


sns.distplot(a = alldata[alldata.train_test == 'test']['Elevation'], label = "test")
sns.distplot(a = alldata[alldata.train_test == 'train']['Elevation'], label = "train")

# Add title
plt.title("Histogram of Elevation, by train or test")

# Force legend to appear
plt.legend()


# The distribution plots are different for the train and the test set. In the train set you see three peaks, whereas there is only one peak in the test set. Lets have a look at the distributions for each type of forest:

# In[ ]:


train['mean_Hillshade'] = (train.Hillshade_3pm + train.Hillshade_Noon + train.Hillshade_9am)/3

# make 7 new train sets, one for each cover type
spruce = train[train.Cover_Type == 1]
lodgepole = train[train.Cover_Type == 2]
ponderosa = train[train.Cover_Type == 3]
cottonwood = train[train.Cover_Type == 4]
aspen = train[train.Cover_Type == 5]
douglas = train[train.Cover_Type == 6]
krummholz = train[train.Cover_Type == 7]

# set figure size
plt.figure(figsize=(14,6))

# make the plots
sns.distplot(a = spruce['Elevation'], label = "Spruce")
sns.distplot(a = lodgepole['Elevation'], label = "Lodgepole")
sns.distplot(a = ponderosa['Elevation'], label = "Ponderosa")
sns.distplot(a = cottonwood['Elevation'], label = "Cottonwood")
sns.distplot(a = aspen['Elevation'], label = "Aspen")
sns.distplot(a = douglas['Elevation'], label = "Douglas")
sns.distplot(a = krummholz['Elevation'], label = "Krummholz")

# Add title
plt.title("Histogram of Elevation, by cover type")

# Force legend to appear
plt.legend()


# It seems that each type of forest prevers a particular Elevation. Cottonwood is mainly found between 2000 and 2500 meters, Aspen is mainly found between 2500 and 3000 meters, krummholz between 3000 and 4000 meters. Ponderosa and Douglas is found between 2000 and almost 3000 meters. Lodgepole is found between 2500 and 3500 meters. Spruce is found between 2500 and 3500 meters. This makes Elevation a good predictor.

# ### 6.1.2 Aspect

# In[ ]:


# set figure size
plt.figure(figsize=(14,6))

sns.distplot(a = spruce['Aspect'], label = 'spruce')
sns.distplot(a = lodgepole['Aspect'], label = 'lodgepole')
sns.distplot(a = ponderosa['Aspect'], label = 'ponderasa')
sns.distplot(a = cottonwood['Aspect'], label = 'cottonwood')
sns.distplot(a = aspen['Aspect'], label = 'aspen')
sns.distplot(a = douglas['Aspect'], label = 'douglas')
sns.distplot(a = krummholz['Aspect'], label = 'krumholz')

# Add title
plt.title("Histogram of Aspect, by cover type")

# Force legend to appear
plt.legend()


# It seems that Aspect follows a non normal distribution. This is because Aspect represents compass directions. We see that there is a peak for cottonwood forrests between 100 and 150 degrees. This means that cottonwoods forrests are more likely to be found on sites that have a south east downhill direction. Douglas forrest are more likely to be found on sites that have a north downhill direction. I also made a new variable based on compass directions.

# In[ ]:


# make new variable compass
def Compass(row):
    if row.Aspect < 22.5:
        return 'N'
    elif row.Aspect < 67.5:
        return 'NE'
    elif row.Aspect < 112.5:
        return 'E'
    elif row.Aspect < 157.5:
        return 'SE'
    elif row.Aspect < 202.5:
        return 'S'
    elif row.Aspect < 247.5:
        return 'SW'
    elif row.Aspect < 292.5:
        return 'W'
    elif row.Aspect < 337.5:
        return 'NW'
    else:
        return 'N'
    
train['Compass'] = train.apply(Compass, axis='columns')


# In[ ]:


df_plot = train.groupby(['CoverType', 'Compass']).size().reset_index().pivot(columns='CoverType', index='Compass', values=0)
df_plot.plot(kind='bar', stacked=True)

# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# Sites with an east direction are more likely to have Aspen or Cottonwood type of forests. Douglas-fir woods are more likely to grow on sites with a North, North-East or North-West direction. Cottonwood/willow forests are also more likely to grow on sites with a South-East direction.

# ### 6.1.3 Slope

# In[ ]:


# set figure size
plt.figure(figsize=(14,6))

sns.distplot(a = spruce['Slope'], label = 'spruce')
sns.distplot(a = lodgepole['Slope'], label = 'lodgepole')
sns.distplot(a = ponderosa['Slope'], label = 'ponderosa')
sns.distplot(a = cottonwood['Slope'], label = 'cottonwood')
sns.distplot(a = aspen['Slope'], label = 'aspen')
sns.distplot(a = douglas['Slope'], label = 'douglas')
sns.distplot(a = krummholz['Slope'], label = 'krummholz')

# Add title
plt.title("Histogram of Slope, by cover type")

# Force legend to appear
plt.legend()


# Its seems that spruce, lodgepole and cottonwood forest are more likely to found on areas with a lower slope. The other forest types can be found on sites with a larger slope.

# ### Horizontal and vertical distance to hydrology

# In[ ]:


# set figure size
plt.figure(figsize=(14,6))


sns.distplot(a = spruce['Horizontal_Distance_To_Hydrology'], label = 'spruce')
sns.distplot(a = lodgepole['Horizontal_Distance_To_Hydrology'], label = 'lodgepole')
sns.distplot(a = ponderosa['Horizontal_Distance_To_Hydrology'], label = 'ponderosa')
sns.distplot(a = cottonwood['Horizontal_Distance_To_Hydrology'], label = 'cottonwood')
sns.distplot(a = aspen['Horizontal_Distance_To_Hydrology'], label = 'aspen')
sns.distplot(a = douglas['Horizontal_Distance_To_Hydrology'], label = 'douglas')
sns.distplot(a = krummholz['Horizontal_Distance_To_Hydrology'], label = 'krummholz')

# Add title
plt.title("Histogram of Horizontal_Distance_To_Hydrology, by cover type")

# Force legend to appear
plt.legend()


# Horizontal distance to hydrology follows a right skewed distribution. It seems that Cottonwood is frequents on sites that are very close to water. 

# In[ ]:


# set figure size
plt.figure(figsize=(14,6))


sns.distplot(a = spruce['Vertical_Distance_To_Hydrology'])
sns.distplot(a = lodgepole['Vertical_Distance_To_Hydrology'])
sns.distplot(a = ponderosa['Vertical_Distance_To_Hydrology'])
sns.distplot(a = cottonwood['Vertical_Distance_To_Hydrology'])
sns.distplot(a = aspen['Vertical_Distance_To_Hydrology'])
sns.distplot(a = douglas['Vertical_Distance_To_Hydrology'])
sns.distplot(a = krummholz['Vertical_Distance_To_Hydrology'])

# Add title
plt.title("Histogram of Vertical_Distance_To_Hydrology, by cover type")

# Force legend to appear
plt.legend()


# Here you also see that cottonwood prefers sites close to water.

# In[ ]:


# set figure size
plt.figure(figsize=(14,6))

sns.boxplot(y="CoverType", x="Horizontal_Distance_To_Hydrology", data=train)


# In[ ]:


# set figure size
plt.figure(figsize=(14,6))

sns.boxplot(y="CoverType", x="Vertical_Distance_To_Hydrology", data=train)


# Maybe it is possible to combine somehow this information.

# In[ ]:


sns.scatterplot(x = 'Vertical_Distance_To_Hydrology', 
                y = 'Horizontal_Distance_To_Hydrology', 
                hue = 'CoverType',
                data = train)


# In[ ]:


# set figure size
plt.figure(figsize=(14,6))


sns.distplot(a = spruce['Horizontal_Distance_To_Roadways'])
sns.distplot(a = lodgepole['Horizontal_Distance_To_Roadways'])
sns.distplot(a = ponderosa['Horizontal_Distance_To_Roadways'])
sns.distplot(a = cottonwood['Horizontal_Distance_To_Roadways'])
sns.distplot(a = aspen['Horizontal_Distance_To_Roadways'])
sns.distplot(a = douglas['Horizontal_Distance_To_Roadways'])
sns.distplot(a = krummholz['Horizontal_Distance_To_Roadways'])

# Add title
plt.title("Histogram of Horizontal_Distance_To_Roadways, by cover type")

# Force legend to appear
plt.legend()


# In[ ]:


# set figure size
plt.figure(figsize=(14,6))

sns.boxplot(y="CoverType", x="Horizontal_Distance_To_Roadways", data=train)


# ### Hillshade index
# 
# Remember that hillshade index ranges from 0 (dark) to 255 (bright). Thus the higher the number, the more shadow.

# In[ ]:


# set figure size
plt.figure(figsize=(14,6))


sns.distplot(a = spruce['Hillshade_9am'], label = "spruce")
sns.distplot(a = lodgepole['Hillshade_9am'], label = "lodgepole")
sns.distplot(a = ponderosa['Hillshade_9am'], label = "ponderosa")
sns.distplot(a = cottonwood['Hillshade_9am'], label = "cottonwood")
sns.distplot(a = aspen['Hillshade_9am'], label = "aspen")
sns.distplot(a = douglas['Hillshade_9am'], label = "douglas")
sns.distplot(a = krummholz['Hillshade_9am'], label = "krummholz")

# Add title
plt.title("Histogram of Hillshade_9am, by cover type")

# Force legend to appear
plt.legend()


# Here you see some seperation for hill_shade index. Cottonwood forests prefer sun (low hillshade index) and douglas and ponderosa prefer sites with less sun (high hillshade index)

# In[ ]:


# set figure size
plt.figure(figsize=(14,6))


sns.distplot(a = spruce['Hillshade_Noon'])
sns.distplot(a = lodgepole['Hillshade_Noon'])
sns.distplot(a = ponderosa['Hillshade_Noon'])
sns.distplot(a = cottonwood['Hillshade_Noon'])
sns.distplot(a = aspen['Hillshade_Noon'])
sns.distplot(a = douglas['Hillshade_Noon'])
sns.distplot(a = krummholz['Hillshade_Noon'])

# Add title
plt.title("Histogram of Hillshade_Noon, by cover type")

# Force legend to appear
plt.legend()


# For hill shade index at noon you don't see a clear seperation.

# In[ ]:


# set figure size
plt.figure(figsize=(14,6))


sns.distplot(a = spruce['Hillshade_3pm'])
sns.distplot(a = lodgepole['Hillshade_3pm'])
sns.distplot(a = ponderosa['Hillshade_3pm'])
sns.distplot(a = cottonwood['Hillshade_3pm'])
sns.distplot(a = aspen['Hillshade_3pm'])
sns.distplot(a = douglas['Hillshade_3pm'])
sns.distplot(a = krummholz['Hillshade_3pm'])

# Add title
plt.title("Histogram of Hillshade_3pm, by cover type")

# Force legend to appear
plt.legend()


# What happens if we make a new variable mean_hillshade

# In[ ]:


train['mean_Hillshade'] = (train.Hillshade_3pm + train.Hillshade_Noon + train.Hillshade_9am)/3


# In[ ]:


# set figure size
plt.figure(figsize=(14,6))


sns.distplot(a = spruce['mean_Hillshade'])
sns.distplot(a = lodgepole['mean_Hillshade'])
sns.distplot(a = ponderosa['mean_Hillshade'])
sns.distplot(a = cottonwood['mean_Hillshade'])
sns.distplot(a = aspen['mean_Hillshade'])
sns.distplot(a = douglas['mean_Hillshade'])
sns.distplot(a = krummholz['mean_Hillshade'])

# Add title
plt.title("Histogram of mean_Hillshade, by cover type")

# Force legend to appear
plt.legend()


# In[ ]:


# set figure size
plt.figure(figsize=(10,10))

# make scatter plot of elevation and mean_Hillshade, color by CoverType
sns.scatterplot(y = train.Elevation, 
                x = train.mean_Hillshade,
                hue = train.CoverType)


# In[ ]:


# make scatter plot of elevation and mean_Hillshade, color by CoverType
sns.scatterplot(y = train.Elevation, 
                x = train.Hillshade_9am,
                hue = train.CoverType)


# In[ ]:


sns.scatterplot(y = train.Elevation, 
                x = train.Hillshade_Noon,
                hue = train.CoverType)


# In[ ]:


sns.scatterplot(y = train.Elevation, 
                x = train.Hillshade_3pm,
                hue = train.CoverType)


# Interesting here you see a nice separation!!!

# In[ ]:


g = sns.FacetGrid(train, col="Compass", hue="CoverType", col_wrap=2)
g.map(plt.scatter, "Elevation", "Hillshade_9am", alpha=.5)
g.add_legend()


# ### Horizontal distance to fire points

# In[ ]:


sns.distplot(a = spruce['Horizontal_Distance_To_Fire_Points'])
sns.distplot(a = lodgepole['Horizontal_Distance_To_Fire_Points'])
sns.distplot(a = ponderosa['Horizontal_Distance_To_Fire_Points'])
sns.distplot(a = cottonwood['Horizontal_Distance_To_Fire_Points'])
sns.distplot(a = aspen['Horizontal_Distance_To_Fire_Points'])
sns.distplot(a = douglas['Horizontal_Distance_To_Fire_Points'])
sns.distplot(a = krummholz['Horizontal_Distance_To_Fire_Points'])

# Add title
plt.title("Histogram of Horizontal_Distance_To_Fire_Points, by cover type")

# Force legend to appear
plt.legend()


# In[ ]:


# set figure size
plt.figure(figsize=(14,6))

sns.boxplot(y="CoverType", x="Horizontal_Distance_To_Fire_Points", data=train)


# In[ ]:


sns.scatterplot(x = 'Horizontal_Distance_To_Fire_Points',
               y = 'Horizontal_Distance_To_Roadways',
                hue = 'CoverType',
               data = train)


# **5.2 Wilderness Area**
# 
# There are four different types of wilderness area: 
# 
# 1. Wilderness_area1: [Rawah Wilderness Area ](https://en.wikipedia.org/wiki/Rawah_Wilderness)
# 2. Wilderness_area2: [Neota Wilderness Area](https://en.wikipedia.org/wiki/Neota_Wilderness)
# 3. Wilderness_area3: [Comanche Peak Wilderness Area](https://en.wikipedia.org/wiki/Comanche_Peak_Wilderness)
# 4. Wilderness_area4: [Cache la Poudre Wilderness Area](https://en.wikipedia.org/wiki/Cache_La_Poudre_Wilderness) 
# 
# 
# 

# In[ ]:


# make new variable Wilderness

def Wilderness(row):
    if row.Wilderness_Area1 == 1:
        return 'Rawah'
    elif row.Wilderness_Area2 == 1:
        return 'Neota'
    elif row.Wilderness_Area3 == 1:
        return 'Comanche Peak'
    elif row.Wilderness_Area4 == 1:
        return 'Cache la Poudre'
    else:
        return 0

train['Wilderness'] = train.apply(Wilderness, axis='columns')
test['Wilderness'] = test.apply(Wilderness, axis='columns')
alldata['Wilderness'] = alldata.apply(Wilderness, axis='columns')


# In[ ]:


plt.figure(figsize=(10,4))

plt.title("Countplot of Wilderness areas in train set")

sns.countplot(x = train.Wilderness)


# The plot above shows that in the train set most observations are from sites found in the Comanche Peak Wilderness area. The next in row is the Cache de la Poudre wilderness area, third in line is the Rawah Wilderness Area. The train set contains relatively few observations from sites in the Neota Wilderness Area. 

# In[ ]:


plt.figure(figsize=(10,4))

plt.title("Countplot of Wilderness in test set")

sns.countplot(x = test.Wilderness)


# In the train set most observations are from sites in the Rawah Wilderness area and the Comanche Peak wilderness area. Observations from sites located in the Cache la Poudre area and the Neota wilderness area are relatively rare. Thus compared to the train set, the Cache de la Poudre is underrepresented and the Rawah Wilderness area is overrepresented.  

# In[ ]:


# get the relevant data
r = [0,1,2,3]

df = train.groupby(['CoverType', 'Wilderness']).size().reset_index().pivot(columns='CoverType', index='Wilderness', values=0)
df = df.fillna(0)

totals = df.sum(axis = 1, skipna = True) 


aspen = [i / j * 100 for i,j in zip(df['Aspen'], totals)]
cottonwood = [i / j * 100 for i,j in zip(df['Cottonwood/Willow'], totals)]
douglas = [i / j * 100 for i,j in zip(df['Douglas-fir'], totals)]
krummholz = [i / j * 100 for i,j in zip(df['Krummholz'], totals)]
lodgepole = [i / j * 100 for i,j in zip(df['Lodgepole Pine'], totals)]
ponderosa = [i / j * 100 for i,j in zip(df['Ponderosa Pine'], totals)]
spruce = [i / j * 100 for i,j in zip(df['Spruce/Fir'], totals)]


# plot
barWidth = 0.85
names = ('Cache la Poudre','Comance Peak','Neota','Rawah')

# Create aspen bars
plt.bar(r, aspen, color='mediumblue', width=barWidth, label="Aspen")
# Create cottonwood bars
plt.bar(r, cottonwood, bottom=aspen, color='darkorange', width=barWidth, label="Cottonwood/Willow")
# Create douglas bars
plt.bar(r, douglas, bottom=[i+j for i,j in zip(aspen, 
                                               cottonwood)], color='forestgreen', width=barWidth, label = "Douglas-Fir")
# Create krummholz bars
plt.bar(r, krummholz, bottom=[i+j+k for i,j,k in zip(aspen, 
                                               cottonwood, 
                                               douglas)], color='red', width=barWidth, label = "Krummholz")

# Create lodgepole bars
plt.bar(r, lodgepole, bottom=[i+j+k+l for i,j,k,l in zip(aspen, 
                                               cottonwood, 
                                               douglas,
                                               krummholz)], color='darkviolet', width=barWidth, label = "Lodgepole Pine")
                            
# Create ponderosa bars
plt.bar(r, ponderosa, bottom=[i+j+k+l+m for i,j,k,l,m in zip(aspen, 
                                                cottonwood, 
                                               douglas,
                                               krummholz,
                                               lodgepole)], color='brown', width=barWidth, label = "Ponderosa Pine")

# Create spruce bars
plt.bar(r, spruce, bottom=[i+j+k+l+m+n for i,j,k,l,m,n in zip(aspen, 
                                               cottonwood, 
                                               douglas,
                                               krummholz,
                                               lodgepole,
                                               ponderosa)], color='violet', width=barWidth, label = "Spruce/Fir")

 
# Custom x axis and y axis
plt.xticks(r, names)
plt.xlabel("Wilderness area")
plt.ylabel("Percentage")

# give title to chart
plt.title("Forest types by Wilderness Area")


# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# Show graphic
plt.show()


# The above plot shows that not all cover types are present in each Wilderness Area. In the Cache de la Poudre Wilderness Area three types are present (Ponderosa Pine, Douglas-Fir and Cottonwood/Willow). In the Comance Peak 5 types are present (Cottonwood/Willow is missing). In the Neota Wilderness area three cover types  are present (Spruce/Fir, lodgepole pine and krummholz). In the Rawah Wilderness four cover types are present (Aspen, Krummholz, Lodgepole Pine, Spruce/Fir). 
# 
# The disbalance for Wilderness area in the train and test set can probably be explained by the observation the the train set was balanced for cover type.

# **5.3 Soil Types**
# 
# There are 40 different soil types. I start with making a new variable that indicates the soil type per site

# In[ ]:


# this is really slooooooow, is there a different to programm this, to make it faster??

def SoilType(row):
    if row.Soil_Type1 == 1:
        return 1
    elif row.Soil_Type2 == 1 :
        return 2
    elif row.Soil_Type3 == 1 :
        return 3
    elif row.Soil_Type4 == 1:
        return 4
    elif row.Soil_Type5 == 1:
        return 5
    elif row.Soil_Type6 == 1:
        return 6
    elif row.Soil_Type7 == 1:
        return 7
    elif row.Soil_Type8 == 1:
        return 8
    elif row.Soil_Type9 == 1:
        return 9
    elif row.Soil_Type10 == 1:
        return 10
    elif row.Soil_Type11 == 1:
        return 11
    elif row.Soil_Type12 == 1:
        return 12
    elif row.Soil_Type13 == 1:
        return 13
    elif row.Soil_Type14 == 1:
        return 14
    elif row.Soil_Type15 == 1:
        return 15
    elif row.Soil_Type16 == 1 :
        return 16
    elif row.Soil_Type17 == 1 :
        return 17
    elif row.Soil_Type18 == 1:
        return 18
    elif row.Soil_Type19 == 1:
        return 19
    elif row.Soil_Type20 == 1:
        return 20
    elif row.Soil_Type21 == 1:
        return 21
    elif row.Soil_Type22 == 1:
        return 22
    elif row.Soil_Type23 == 1:
        return 23
    elif row.Soil_Type24 == 1:
        return 24
    elif row.Soil_Type25 == 1:
        return 25
    elif row.Soil_Type26 == 1:
        return 26
    elif row.Soil_Type27 == 1:
        return 27
    elif row.Soil_Type28 == 1:
        return 28
    elif row.Soil_Type29 == 1:
        return 29
    elif row.Soil_Type30 == 1:
        return 30
    elif row.Soil_Type31 == 1:
        return 31
    elif row.Soil_Type32 == 1:
        return 32
    elif row.Soil_Type33 == 1:
        return 33
    elif row.Soil_Type34 == 1:
        return 34
    elif row.Soil_Type35 == 1:
        return 35
    elif row.Soil_Type36 == 1:
        return 36
    elif row.Soil_Type37 == 1:
        return 37
    elif row.Soil_Type38 == 1:
        return 38
    elif row.Soil_Type39 == 1:
        return 39
    elif row.Soil_Type40 == 1:
        return 40
    else:
        return 0
    
train['SoilType'] = train.apply(SoilType, axis='columns')
test['SoilType'] = test.apply(SoilType, axis='columns')
#alldata['SoilType'] = alldata.apply(SoilType, axis='columns')


# In[ ]:


# set figure size
plt.figure(figsize=(14,6))

#add title
plt.title("Countplot of SoilType")

sns.countplot(x = train.SoilType)


# Here we already see that some soil types are very rare. These very rare soil type will not add much value when predicting forest types and should be removed when trying to predict forest type.

# In[ ]:


# set figure size
plt.figure(figsize=(14,6))

#add title
plt.title("Countplot of SoilType")

sns.countplot(x = test.SoilType)


# In[ ]:



# get the relevant data
r = list(range(1,39))

df = train.groupby(['CoverType', 'SoilType']).size().reset_index().pivot(columns='CoverType', index='SoilType', values=0)
df = df.fillna(0)

totals = df.sum(axis = 1, skipna = True) 


aspen = [i / j * 100 for i,j in zip(df['Aspen'], totals)]
cottonwood = [i / j * 100 for i,j in zip(df['Cottonwood/Willow'], totals)]
douglas = [i / j * 100 for i,j in zip(df['Douglas-fir'], totals)]
krummholz = [i / j * 100 for i,j in zip(df['Krummholz'], totals)]
lodgepole = [i / j * 100 for i,j in zip(df['Lodgepole Pine'], totals)]
ponderosa = [i / j * 100 for i,j in zip(df['Ponderosa Pine'], totals)]
spruce = [i / j * 100 for i,j in zip(df['Spruce/Fir'], totals)]


# plot
barWidth = 0.3
names = list(range(1,39)) # is not correct!!!!

# set figure size
plt.figure(figsize=(12,6))

# Create aspen bars
plt.bar(r, aspen, color='mediumblue', width=barWidth, label="Aspen")

# Create cottonwood bars
plt.bar(r, cottonwood, bottom=aspen, color='darkorange', width=barWidth, label="Cottonwood/Willow")

# Create douglas bars
plt.bar(r, douglas, bottom=[i+j for i,j in zip(aspen, 
                                               cottonwood)], color='forestgreen', width=barWidth, label = "Douglas-Fir")
# Create krummholz bars
plt.bar(r, krummholz, bottom=[i+j+k for i,j,k in zip(aspen, 
                                               cottonwood, 
                                               douglas)], color='red', width=barWidth, label = "Krummholz")

# Create lodgepole bars
plt.bar(r, lodgepole, bottom=[i+j+k+l for i,j,k,l in zip(aspen, 
                                               cottonwood, 
                                               douglas,
                                               krummholz)], color='darkviolet', width=barWidth, label = "Lodgepole Pine")
                            
# Create ponderosa bars
plt.bar(r, ponderosa, bottom=[i+j+k+l+m for i,j,k,l,m in zip(aspen, 
                                                cottonwood, 
                                               douglas,
                                               krummholz,
                                               lodgepole)], color='brown', width=barWidth, label = "Ponderosa Pine")

# Create spruce bars
plt.bar(r, spruce, bottom=[i+j+k+l+m+n for i,j,k,l,m,n in zip(aspen, 
                                               cottonwood, 
                                               douglas,
                                               krummholz,
                                               lodgepole,
                                               ponderosa)], color='violet', width=barWidth, label = "Spruce/Fir")

 
# Custom x axis and y axis
plt.xticks(r, names)
plt.xlabel("Soil type")
plt.ylabel("Percentage")

# give title to chart
plt.title("Forest types by soil type")


# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# Show graphic
plt.show()


# The above plot shows that soiltype is important to predict which kind of forest is growing on which site. For examples on sites with soiltype 3, Cottonwood/willow is the dominant species. On soiltypes 35-40, it seems that krumholz forests are dominant. 
# 
# In the train set there are two soiltypes that are not present!!!
