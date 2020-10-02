#!/usr/bin/env python
# coding: utf-8

# # Loading the libraries

# In[1]:


# Load libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# # Loading and Exploring the training data

# In[2]:


# Load data

df = pd.read_csv('../input/train.csv')
df.info(verbose=False)


# In[3]:


# Column names

print(len(df.columns))
df.columns


# # Dimention reduction 
# ----
#  - Reducing the **40 Soil Type columns into a single Soil Type** column and 
#  - **4 Wilderness Areas ino a single Wilderness Area** column.

#  Creating list to map to the various Soil Types and Wilderness Areas

# In[4]:


# Soil_Type and Wilderness_Area list

soil_list = []
for i in range(1, 41):
    soil_list.append('Soil_Type' + str(i))

wilderness_area_list = ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']

print(soil_list, "\n")
print(wilderness_area_list)


# Checking for null values in the data

# In[5]:


# Check for null values

print("Total number of NaN values: ", df.isnull().sum().sum())


# Checking is any single row is mapped to more that one Wilderness area
# 
# As there is no multiple mapping, combine the features into a single column maintaining the required mapping

# In[6]:


# Check for multiple Wilderness_Area for a single Cover_Type

wilderness_area_test = df[wilderness_area_list].sum(axis=1)
wilderness_area_test.unique()


# Checking is any single row is mapped to more that one Soil Type
# 
# As there is no multiple mapping, combine the features into a single column maintaining the required mapping

# In[7]:


# Check for multiple Soil_Types for a single Cover_Type

soil_type_test = df[soil_list].sum(axis=1)
soil_type_test.unique()


# # Compress Wilderness Areas
# ----
# 
# 1 - Rawah Wilderness Area
# 
# 
# 2 - Neota Wilderness Area
# 
# 
# 3 - Comanche Peak Wilderness Area
# 
# 
# 4 - Cache la Poudre Wilderness Area
# 
# ----
# Compressing the features in to a single column** 'Wilderness Area'** using the given mapping

# In[8]:


# Wilderness areas

def wilderness_compress(df):
    
    df[wilderness_area_list] = df[wilderness_area_list].multiply([1, 2, 3, 4], axis=1)
    df['Wilderness_Area'] = df[wilderness_area_list].sum(axis=1)
    return df


# # Compress Soil Types
# ----
# 
# 1 Cathedral family - Rock outcrop complex, extremely stony.
# 
# 2 Vanet - Ratake families complex, very stony.
# 
# . . .
# 
# 39 Moran family - Cryorthents - Leighcan family complex, extremely stony.
# 
# 40 Moran family - Cryorthents - Rock land complex, extremely stony.
# 
# ----
# Compressing the features in to a single column** 'Soil Type'** using the given mapping

# In[9]:


# Soil types

def soil_compress(df):
    
    df[soil_list] = df[soil_list].multiply([i for i in range(1, 41)], axis=1)
    df['Soil_Type'] = df[soil_list].sum(axis=1)
    return df


# In[10]:


# Compressing features

df = wilderness_compress(df)
df = soil_compress(df)

df[['Wilderness_Area', 'Soil_Type']].head()


#  Cerating a list of the useful columns of the training data
# 
#  Removing all **Wilderness Areas** and **Soil Types**

# In[11]:


# Useful columns

cols = df.columns.tolist()
columns = cols[1:11] + cols[56:]

print("Useful columns: ", columns)

values = df[columns]
labels = df['Cover_Type']

print("Values: ", values.shape)
print("Labels: ", labels.shape)


# # Exploratory Data Analysis

# In[12]:


# Set style

sns.set_style('whitegrid')
sns.set(rc={'figure.figsize':(11.7,8.27)})


# ####  Cover Type distribution in training data
# ----
# Perfectly distributed dataset

# In[13]:


# Cover Type countplot

ax = sns.countplot(labels, alpha=0.75)
ax.set(xlabel='Cover Type', ylabel='Number of labels')
plt.show()


# ####  Disrtibution of Elevation

# In[14]:


# Elevation distribution

ax = sns.distplot(df['Elevation'], color='pink')
plt.show()


# ####  Disrtibution of Aspect

# In[15]:


# Aspect distribution

ax = sns.distplot(df['Aspect'], color='#add8e6')
plt.show()


# ####  Disrtibution of Slope

# In[16]:


# Slope distribution

ax = sns.distplot(df['Slope'], color='#90ee90')
plt.show()


# ####  Distance to Hydrology with Elevation

# In[17]:


# Distance to hydrology (Horizontal versus Vertical) with Elevation

ax = plt.scatter(x=df['Horizontal_Distance_To_Hydrology'], y=df['Vertical_Distance_To_Hydrology'], c=df['Elevation'], cmap='jet')
plt.xlabel('Horizontal Distance')
plt.ylabel('Vertical Distance')
plt.title("Distance to Hydrology with Elevation")
plt.show()


# ####  Disrtibution of Hillshade at 9am

# In[18]:


# Hillshade at 9am distribution

ax = sns.distplot(df['Hillshade_9am'], color='#fcd14d')
plt.show()


# ####  Disrtibution of Hillshade at Noon

# In[19]:


# Hillshade at noon distribution

ax = sns.distplot(df['Hillshade_Noon'], color='#fdb813')
plt.show()


# ####  Disrtibution of Hillshade at 3pm

# In[20]:


# Hillshade at 3pm distribution

ax = sns.distplot(df['Hillshade_3pm'], color='orange')
plt.show()


# #### Wilderness Area distribution in training data

# In[21]:


# Wilderness area countplot

ax = sns.countplot(df['Wilderness_Area'], alpha=0.75)
ax.set(xlabel='Wilderness Area Type', ylabel='Number of Areas', title='Wilderness areas - Count')
plt.show()


# #### Distribution of Wilderness Area with respect to Cover Type

# In[22]:


# Wilderness area to Cover type mapping

ax = plt.scatter(x=df['Wilderness_Area'], y=df['Cover_Type'], c=df['Wilderness_Area'], cmap='Set2', s=500, marker='s', alpha=0.01)
plt.xlabel('Wilderness Area Type')
plt.xticks([1, 2, 3, 4])
plt.ylabel('Cover Type')
plt.title("Wilderness Area versus Cover Type")
plt.show()


# #### Soil Type  distribution in training data

# In[23]:


# Soil Type countplot

ax = sns.countplot(df['Soil_Type'], alpha=0.75)
ax.set(xlabel='Soil Type', ylabel='Count', title='Soil types - Count')
plt.show()


# #### Distribution of Soil Type with respect to Cover Type

# In[24]:


# Soil Type jointplot

ax = sns.jointplot(x='Soil_Type', y='Cover_Type', data=df, kind='kde', color='purple')
plt.show()


# # Cleaned training dataset with Dimention Reduction

# In[25]:


# Reduced dataset

clean = df[['Id', 'Cover_Type'] + columns]
clean.head()


# In[26]:


# Clean data columns

print(len(clean.columns))
clean.columns


# In[27]:


# CSV file

clean.to_csv('clean_train.csv', index=False)


# In[ ]:




