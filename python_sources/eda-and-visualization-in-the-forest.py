#!/usr/bin/env python
# coding: utf-8

# Based on the notebook:
# https://www.kaggle.com/fatihbilgin/quick-visualization-and-eda-for-beginners

# *1. Importing Libraries*

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pp
import plotly.graph_objs as go
from plotly.offline import iplot
import plotly.express as px

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# *2.  Exploratory Data Analysis (EDA)*

# In[ ]:


df_test = pd.read_csv('../input/learn-together/test.csv')
df_train = pd.read_csv('../input/learn-together/train.csv')
df_train.head()


# In[ ]:


print("Train dataset shape: "+ str(df_train.shape))
print("Test dataset shape:  "+ str(df_test.shape))


# In[ ]:


df_train.info()


# There are a total of 56 columns in the train dataset (55 in test dataset except the last column that is y), All of them are not null.

# In[ ]:


df_train.describe().T


# All Wilderness_Area and Soil_Type columns have values in the range of 0 and 1. Quite likely these columns are categorical and consist of 0 and 1. To validate this let's check distinct values of the following columns:

# In[ ]:


print(df_train.iloc[:,10:-1].columns)


# In[ ]:


pd.unique(df_train.iloc[:,10:-1].values.ravel())


# In[ ]:


df_train.iloc[:,10:-1] = df_train.iloc[:,10:-1].astype("category")
df_test.iloc[:,10:] = df_test.iloc[:,10:].astype("category")


# In[ ]:


f,ax = plt.subplots(figsize=(8,6))
sns.heatmap(df_train.corr(),annot=True, linewidths=.5, fmt='.1f', cmap="viridis", ax=ax)
plt.show()


# It seems the most important correlations are between "Horizontal Distance To Hydrology" and "Vertical Distance To Hydrology" with 70%; between "Aspect" and "Hillshade 3pm" with 60%; between "Hillshade Noon" and "Hillshade 3pm" with %60; between "Elevation" and "Horizontal Distance To Roadways" with %60. Let's see how they look in a scatter plot.
# 
# 

# *3.  Data Visualization*

# In[ ]:


df_train.plot(kind='scatter', x='Vertical_Distance_To_Hydrology', y='Horizontal_Distance_To_Hydrology', alpha=0.6, color='blue', figsize = (12,8))
plt.title('Vertical And Horizontal Distance To Hydrology')
plt.xlabel("Vertical Distance")
plt.ylabel("Horizontal Distance")
plt.show()


# In[ ]:


df_train.plot(kind='scatter', x='Aspect', y='Hillshade_3pm', alpha=0.6, color='purple', figsize = (12,8))
plt.title('Aspect and Hillshade 3pm Relation')
plt.xlabel("Aspect")
plt.ylabel("Hillshade 3pm")
plt.show()


# In[ ]:


df_train.plot(kind='scatter', x='Hillshade_Noon', y='Hillshade_3pm', alpha=0.9, color='teal', figsize = (12,9))
plt.title('Hillshade Noon and Hillshade 3pm Relation')
plt.xlabel("Hillshade_Noon")
plt.ylabel("Hillshade 3pm")
plt.show()


# There are obvious patterns we can use to fit to our model.  We just need to ignore the outliers.  With Boxplots we can best visualize them.

# In[ ]:


ax = sns.boxplot(data=df_train.iloc[:,4:6], palette="BrBG")
ax.set_ylabel('Distance (meters)')


# In[ ]:


ax = sns.boxplot(data=df_train.iloc[:,8:10], palette="PRGn")
ax.set_ylabel('Aspect in degrees azimuth (0-250)')


# Ok great, lets look at some histograms to compare Horizontal Distance to Hydrology and Vertical Distance to Hydrology.

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(15,7))
df_train.Horizontal_Distance_To_Hydrology.plot.hist(ax=ax[0],bins=30,edgecolor='black',color='darkorange')
ax[0].set_title('Horizontal Distance To Hydrology')
x1=list(range(0,1000,100))
ax[0].set_xticks(x1)
ax[0].set_xlabel('distance (meters)')
ax[0].set_ylabel('Total Count Observed Trees')
df_train.Vertical_Distance_To_Hydrology.plot.hist(ax=ax[1],bins=30,edgecolor='black',color='C')
ax[1].set_title('Vertical Distance To Hydrology')
x2=list(range(-150,350,50))
ax[1].set_xticks(x2)
ax[1].set_xlabel('distance (meters)')
ax[1].set_ylabel('Total Count Observed Trees')
plt.show()


# Definitely a trend of observed tree counts and proximity to water. Let's now look at soil types.

# In[ ]:


soil_types = df_train.iloc[:,15:-1].sum(axis=0)

plt.figure(figsize=(18,9))
sns.barplot(x=soil_types.index, y=soil_types.values, palette="BrBG")
plt.xticks(rotation= 75)
plt.xlabel('Soil Type')
plt.ylabel('Total Count Observed Trees')
plt.title('Count of Soil Types With Value 1',color = 'darkred',fontsize=12)


# Soil Types of "10" and "29" contain the greatest counts of trees, whereas soil types like "7, 8, 15, and 25" have little to no tree data.  Let's have a look at Wilderness Areas.

# In[ ]:


wilderness_areas = df_train.iloc[:,11:15].sum(axis=0)

plt.figure(figsize=(7,5))
sns.barplot(x=wilderness_areas.index,y=wilderness_areas.values, palette="Blues_d")
plt.xticks(rotation=60)
plt.title('Wilderness Areas',color = 'darkred',fontsize=12)
plt.ylabel('Total Count Observed Trees')
plt.show()


# There are varying quantities of the 15,120 observed trees in both soil type and wilderness area. Let us see about the quantities in the cover type, which is what we will be predicting in the test set.

# In[ ]:


cover_type = df_train["Cover_Type"].value_counts()
df_cover_type = pd.DataFrame({'CoverType': cover_type.index, 'Total':cover_type.values})

fig = px.bar(df_cover_type, x='CoverType', y='Total', height=400, width=650)
fig.show()


# There are equal counts of all 7 cover types in the training set.  Let us take a look at how the 7 cover types relate to Horizontal distance to Roadways, and Horizontal Distance to Hydrology.

# In[ ]:


f, ax=plt.subplots(1,2,figsize=(21,7))
df_train.plot(kind='scatter', ax=ax[0],x='Cover_Type', y='Horizontal_Distance_To_Hydrology', alpha=0.5, color='purple')
ax[0].set_title('Horizontal Distance To Hydrology')
x1=list(range(1,8,1))
ax[0].set_ylabel("distance (meters)")
ax[0].set_xlabel("Cover Type")
df_train.plot(kind='scatter', ax=ax[1],x='Cover_Type', y='Horizontal_Distance_To_Roadways', alpha=0.5, color='purple')
ax[1].set_title('Horizontal Distance To Roadways')
x2=list(range(1,8,1))
ax[1].set_ylabel("")
ax[1].set_xlabel("Cover Type")

plt.show()


# Similar patterns!

# Enjoyed working on this notebook.  Thank you again Fatih Bilgin for your prior work.  It helped me greatly to practice doing Exploratory Data Analysis and make great looking visualizations for this data set.  Looking forward to diving into machine learning to see what we can come up with.
