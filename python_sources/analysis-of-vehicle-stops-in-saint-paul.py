#!/usr/bin/env python
# coding: utf-8

# ## Imports and Data
# 
# A list of using libraries for analysis are below.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # for plotting
import matplotlib.pyplot as plt # for plotting
import folium

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# We cannot see any data, go on deeply!

# In[ ]:


for f in os.listdir("../input/cpe-data"):
    print(f)


# Each folder contains other folders and files. But the main data files (police reports) lie on a level below. Let us check it together. We can see that directories and files name give for us first insight about data. Also, I printed the dimension of data.

# In[ ]:


root = "../input/cpe-data"
for obj in os.listdir(root):
    if os.path.isdir("{}/{}".format(root, obj)):
        print("Folder {} has next data files:".format(obj))
        for file in os.listdir("{}/{}".format(root, obj)):
            if os.path.isfile("{}/{}/{}".format(root, obj,file)):
                print("    -- {} dimension of {}".format(pd.read_csv("{}/{}/{}".format(root, obj, file), skiprows=1).shape, file))                


# ## Part 1. Vehicle Stops

# ### Data

# At first, I most interested in the biggest dataset. The dataset represents information about Vehicle Stops by policemen. As you can see below we needed to skip one of the the first two rows with `skiprows` and also we can parse date field wit `parse_dates` arguments of `read_csv`.  I skipped the second row. The first seems more human.
# As result, we have a table with 710471 records and 12 different columns.

# In[ ]:


file_path = '../input/cpe-data/Dept_24-00098/24-00098_Vehicle-Stops-data.csv'
pd.read_csv(file_path).head()


# In[ ]:


df = pd.read_csv('../input/cpe-data/Dept_24-00098/24-00098_Vehicle-Stops-data.csv', skiprows=(1,2),  parse_dates=['INCIDENT_DATE'])
df.head()


# In[ ]:


df.shape


# In[ ]:


df.columns


# In[ ]:


df.info()


# ### Find Duplicates

# I checked dataset and find some amount duplicates. Not a lot, only 1512.

# In[ ]:


df[df.duplicated(keep='first')].head(6)


# In[ ]:


df[df.duplicated(keep='first')].shape


# In[ ]:


df.drop_duplicates(keep='first', inplace=True)
df = df.reset_index()


# In[ ]:


df.shape


# ### How is distributed of Vehicle Stops by Policemen across years and months?

# In[ ]:


df_pivot = df.pivot_table('INCIDENT_DATE', 
                          index = df['INCIDENT_DATE'].dt.year.rename('YEAR'),
                          columns = df['INCIDENT_DATE'].dt.month.rename('MONTH'), 
                          aggfunc='count')


# In[ ]:


plt.figure(figsize=(15,15))
sns.heatmap(df_pivot, cmap='RdYlGn', fmt="d", annot=True, linewidths=.5)


# ### Where Policemen were stopped of Vehicles at 1 January 2015?

# In[ ]:


print(df['LOCATION_LATITUDE'].min(), df['LOCATION_LATITUDE'].max())
print(df['LOCATION_LONGITUDE'].min(), df['LOCATION_LONGITUDE'].max())


# In[ ]:


df[df['LOCATION_LATITUDE'].isna()].shape


# In[ ]:


df_2015 = df[(df['INCIDENT_DATE'].dt.year==2015)&(df['INCIDENT_DATE'].dt.month==1)&(df['INCIDENT_DATE'].dt.day==1)]
df_2015.shape


# In[ ]:


kmap = folium.Map([44.89, -93.00], height=800, zoom_start=10, tiles='CartoDB dark_matter')
for j, rown in df_2015.iterrows():
    if str(rown["LOCATION_LATITUDE"]) != "nan":
        lon = float(rown["LOCATION_LATITUDE"])
        lat = float(rown["LOCATION_LONGITUDE"])
        folium.CircleMarker([lon, lat], radius=5, color='red', fill=True).add_to(kmap)
kmap


# ### How age of drivers is distributed by race and gender?

# In[ ]:


race_dict = {'Native Am':'Native American'}
df['SUBJECT_RACE'] = df['SUBJECT_RACE'].replace(race_dict)


# In[ ]:


plt.figure(figsize=(16,12))
sns.boxplot(y="SUBJECT_AGE", x="SUBJECT_RACE", hue="SUBJECT_GENDER", data=df)


# ### Conclusions
# It is not the end. To be continued... 

# In[ ]:




