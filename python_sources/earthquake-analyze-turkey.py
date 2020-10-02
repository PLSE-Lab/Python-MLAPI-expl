#!/usr/bin/env python
# coding: utf-8

# ## Eartquake 

# * Goal :Analyze the eartquakes 
# * Dataset:earthquake.csv
# 
# * Notebook will be updated...

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# In[ ]:


earthquake_df= pd.read_csv(("../input/earthquake.csv"))


# In[ ]:


e_df=earthquake_df.copy()


# ## Columns of Dateset

# * ID --> ID of eartquake
# * Date --> Date of the earthquake
# * Time --> Time of earthquake
# * Lat --> Latitude of the earthquake
# * Long --> Longitude of the earthquake
# * Country --> Country of the earthquake
# * City --> City of the earthquake
# * Area --> Area of the earthquake
# * Direction --> Direction of the earthquake
# * Dist --> Distance of direction in KM
# * Depth --> Depth of the earthquake
# * xm -->  Biggest magnitude value in specified magnitude values
# * md -->  Depending on time magnitude
# * Richter --> intensity of earthquake (Richter)
# * mw --> moment magnitude
# * ms --> surface-wave magnitude
# * mb --> body-wave magnitude
# 

# In[ ]:


# Columns
e_df.columns


# In[ ]:


# Info about dataset
e_df.info()


# In[ ]:


# 20 Observation from dataset
e_df.sample(20)


# In[ ]:


# Earthquake Frequency by countries..


# In[ ]:


sns.set_palette(sns.dark_palette('blue',15))
e_df['country'].value_counts().plot(kind='bar',figsize=(30,30),fontsize = 10)
plt.xlabel("Country",fontsize=20,color="Black")
plt.ylabel("Frequency",fontsize=20,color="Black")
plt.show()


# In[ ]:


# Earthquake Frequency for cities of Turkey.


# In[ ]:


sns.set_palette('pastel',15)
e_df['city'].value_counts().plot(kind = "bar", figsize=(30,30),fontsize = 20)
plt.xlabel("City",fontsize=20,color="Black")
plt.ylabel("Frequency",fontsize=20,color="Black")
plt.show()


# In[ ]:


# Plotting Richter's of Earthquakes.


# In[ ]:


sns.set_palette('hot',15)
e_df['city'].value_counts().plot(kind='bar',figsize=(30,30),fontsize = 10)
plt.xlabel("City",fontsize=20,color="Black")
plt.ylabel("Rihcter",fontsize=20,color="Black")
plt.show()



# In[ ]:


sns.set_palette('pastel',15)
e_df['city'].value_counts().plot(kind = "bar", figsize=(30,30),fontsize = 20)
plt.xlabel("City",fontsize=20,color="Black")
plt.ylabel("Frequency",fontsize=20,color="Black")
plt.show()


# In[ ]:


# Creating Correlation Heatmap


# In[ ]:


plt.figure(figsize=(20,20))
sns.heatmap(e_df.corr(), annot = True, fmt= ".1f", linewidths = .3,cmap='coolwarm_r')
plt.show()


# In[ ]:


e_df.city.isin(["istanbul","mugla","van","canakkale","izmir","kutahya","manisa","denizli"])


# # Small-Scale Earthquakes

# ## If Richter between (2.5-5.0) and Depth is > 30 KM 
# ## It's called Small-Scale Earthquakes

# In[ ]:


small_scale = e_df[e_df.city.isin(["mugla","van","canakkale","izmir","kutahya","manisa","denizli","istanbul"]) & (( e_df.richter <= 5.0) & (e_df.richter >=2.5)) & (e_df.depth > 30)]


# In[ ]:


small_scale.head()


# In[ ]:


sns.relplot(x="city", y="richter", kind="scatter",data=small_scale.sample(n=25),hue="date");
plt.xlabel("City",fontsize=20,color="Black")
plt.ylabel("Richter",fontsize=20,color="Black")
plt.title("Some of the Small-Scaled Earthquakes with dates")
plt.show()


# # We've some cities which has suffered more than one big earthquake (richter > 5.5)

# ## We called these earthquakes "Moderate" because of their 'depths'.
# ## If depth is > 15 KM the earthquake is "Moderate"

# In[ ]:


moderate = e_df[e_df.city.isin(["sakarya","kocaeli","mugla","van","izmir","kutahya","denizli","istanbul"]) & ( e_df.richter >= 5.5) & (e_df.depth > 15)]


# In[ ]:


moderate


# In[ ]:


sns.relplot(x="city", y="richter", kind="scatter",data=moderate,hue="date");
plt.xlabel("City",fontsize=20,color="Black")
plt.ylabel("Richter",fontsize=20,color="Black")
plt.title("Some of the Moderate Earthquakes with dates")
plt.show()


# # Large-Scaled Earthquakes
# # If Richter is > 5.5 and Depth is < 12 KM then It's Called Large-Scaled Earthquake

# In[ ]:


large_scaled = e_df[e_df.city.isin(["kocaeli","sakarya","mugla","van","izmir","kutahya","denizli","istanbul"]) & ( e_df.richter >= 5.5) & (e_df.depth < 12)]


# In[ ]:


large_scaled


# In[ ]:


sns.relplot(x="city", y="richter", kind="scatter",data=large_scaled,hue="date");
plt.xlabel("City",fontsize=20,color="Black")
plt.ylabel("Richter",fontsize=20,color="Black")
plt.title("Some of the Large-Scaled Earthquakes with dates")
plt.show()

