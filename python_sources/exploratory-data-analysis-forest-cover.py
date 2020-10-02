#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/train.csv')
df.columns


# In[ ]:


df2 = df.copy()


# In[ ]:


# add target names for readability
def forest(x):
    if x==1:
        return 'Spruce/Fir'
    elif x==2:
        return 'Lodgepole Pine'
    elif x==3:
        return 'Ponderosa Pine'
    elif x==4:
        return 'Cottonwood/Willow'
    elif x==5:
        return 'Aspen'
    elif x==6:
        return 'Douglas-fir'
    elif x==7:
        return 'Krummholz'
    
df2['Cover_Type'] = df2['Cover_Type'].apply(lambda x: forest(x))


# ### Histogram distribution of continuous features

# In[ ]:


df2[df.columns[1:11]].hist(figsize=(20,15),bins=50)
plt.tight_layout()


# ### Check Counts for Target Variable

# In[ ]:


df2['Cover_Type'].value_counts()


# In[ ]:


cmap = sns.color_palette("Set2")

sns.countplot(x='Cover_Type', data=df2, palette=cmap);
plt.xticks(rotation=45);


# Each target forest cover have equal sample sizes

# ### Check Counts for Wilderness & Soil Types

# In[ ]:


# convert dummies into a single column
# convert wilderness
wild_dummies = df[['Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4']]
wild = wild_dummies.idxmax(axis=1)
wild.name = 'Wilderness'

# convert soil
soil_dummies = df[df.columns[15:55]]
soil = soil_dummies.idxmax(axis=1)
soil.name = 'Soil'

# create new dataframe with only cover type, wilderness, soil
wild = pd.concat([df2['Cover_Type'],wild,soil], axis=1)
wild.head()


# In[ ]:


# name wilderness type for readability
def wild_convert(x):
    if x == 'Wilderness_Area1':
        return 'Rawah'
    elif x=='Wilderness_Area2':
        return 'Neota'
    elif x=='Wilderness_Area3':
        return 'Comanche Peak'
    elif x=='Wilderness_Area4':
        return 'Cache la Poudre'
    
wild['Wilderness'] = wild['Wilderness'].apply(lambda x: wild_convert(x))


# In[ ]:


sns.countplot(x='Wilderness',data=wild, palette=cmap);


# Wilderness Area 2, or Neota, is not well represented.

# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot(x='Cover_Type',data=wild, hue='Wilderness');
plt.xticks(rotation=45);


# There appears to be some pattern here, with certain forest cover preferring certain wilderness.

# In[ ]:


plt.figure(figsize=(12,5))
sns.countplot(x='Soil',data=wild);
plt.xticks(rotation=90);


# In[ ]:


# take a look at soil type summary statistics
df[df.columns[15:55]].describe().T.sort_values(by='mean')


# Some soil types like __25 & 8__ only contain 1 sample size. __Soil type 7 & 15__ does not even have 1 sample, hence did not appear in the barplot!

# ### Boxplots, comparsion with forest cover types

# In[ ]:


# select features which are continuous
boxplots = df2[['Elevation', 'Aspect', 'Slope',
               'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
               'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
               'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points','Cover_Type']]


# In[ ]:


cmap = sns.color_palette("Set2")

fig, axes = plt.subplots(ncols=2, nrows=5, figsize=(10, 18))
a = [i for i in axes for i in i]
for i, ax in enumerate(a):
    sns.boxplot(x='Cover_Type', y=boxplots.columns[i], data=boxplots, palette=cmap, width=0.5, ax=ax);

# rotate x-axis for every single plot
for ax in fig.axes:
    plt.sca(ax)
    plt.xticks(rotation=45)

# set spacing for every subplot, else x-axis will be covered
plt.tight_layout()


# We can see that elevation has very different interquantile ranges for each cover type.

# ### Correlation Plots

# In[ ]:


# calculate pearson's correlation, exclude ID
corr = df[df.columns[1:]].corr()

corr['Cover_Type'].sort_values(ascending=False)


# In[ ]:


plt.figure(figsize=(15, 8))
sns.heatmap(corr, cmap=sns.color_palette("RdBu_r", 20));


# Soil Type 7 & Soil Type 15 has no correlation to cover type. 
