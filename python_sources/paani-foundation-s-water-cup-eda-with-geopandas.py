#!/usr/bin/env python
# coding: utf-8

# Notebook submitted in response to task#2: https://www.kaggle.com/bombatkarvivek/paani-foundations-satyamev-jayate-water-cup/tasks?taskId=314 
# 
# Aim is to enrich the WaterCup dataset by joining it with geolocations dataset using 'geopandas' lib. 
# 

# In[ ]:


import geopandas as gpd
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:



# # Read in the data
# full_data = gpd.read_file("../input/maharashtra-location-v2/maharashtra_location.shp")

# # View the first five rows of the data
# # (2903, 7)
# type(full_data)
# data = full_data.loc[:, ["NAME", "geometry"]].copy()
# data.NAME.value_counts()
# data.plot()
# ax = data.plot( figsize=(10,10), color='none',  zorder=3, alpha=0.5, edgecolor='k')
# data.plot(color='lightgreen', ax=ax)


# Load dataset with all villages of Maharashtra along with geo cordinates

# In[ ]:


df_mh_all_villages = gpd.read_file('../input/mh-villages-v2w2/MH_Villages v2W2.shp')


# In[ ]:


# df_mh.plot(figsize=(20,20),zorder=3, alpha=0.5, edgecolor='k')


# In[ ]:


df_mh_all_villages.head()
# (48926, 15)


# In[ ]:


# District wise plot
fig, ax = plt.subplots(1,1,figsize=(20,20))
ax.set_title('Map of all districts of Maharashtra')
df_mh_all_villages.plot(
                        column='DTNAME', 
                        ax=ax,
                        legend=True, 
                        )


# Villages paritcipated in water cup

# In[ ]:


import pandas as pd
df_ListOfTalukas = pd.read_csv("../input/paani-foundations-satyamev-jayate-water-cup/ListOfTalukas.csv")
df_ListOfTalukas.T
# df_ListOfTalukas.isnull().sum()
# MarkingSystem = pd.read_csv("../input/paani-foundations-satyamev-jayate-water-cup/MarkingSystem.csv")
# StateLevelWinners = pd.read_csv("../input/paani-foundations-satyamev-jayate-water-cup/StateLevelWinners.csv")
# VillagesSupportedByDonationsWaterCup2019 = pd.read_csv("../input/paani-foundations-satyamev-jayate-water-cup/VillagesSupportedByDonationsWaterCup2019.csv")


# In[ ]:


df_ListOfTalukas.shape


# In[ ]:


df_mh_all_villages[['DTNAME','GPNAME','VILLNAME']].isnull().sum()


# Some data preprocessing to make both datasets match.

# In[ ]:


_df_ListOfTalukas = df_ListOfTalukas[['District','Taluka']]
# Special case handlings
# Dhule, Nashik, Ahmadnagar
_df_ListOfTalukas = _df_ListOfTalukas.replace('Ahmednagar','Ahmadnagar')                                         .replace('Buldhana','Buldana')                                         .replace('Sangli','Sangali')                                         .replace('Nashik','Nasik')
_df_ListOfTalukas.T
# _df_mh_all_villages[_df_mh_all_villages['DTNAME'].str.contains('Nas')]


# In[ ]:


_df_mh_all_villages = df_mh_all_villages[['DTNAME','GPNAME','VILLNAME','geometry']]
_df_mh_all_villages.T


# Join datasets on 'District' level

# In[ ]:


df_new = pd.merge(_df_mh_all_villages, 
                  _df_ListOfTalukas, 
                  how='outer',
                  left_on=['DTNAME'], 
                  right_on=['District'],
                  indicator=True)


# In[ ]:


_df_new = df_new.replace('both','Participated')                 .replace('left_only','Not Participated')
_df_new['_merge'].unique()


# In[ ]:


_df_new.T


# In[ ]:



fig,ax = plt.subplots(1,1,figsize=(20,20))
ax.set_title('Districts participated in WaterCup.')
_df_new.plot(column='_merge',
            ax=ax,
            legend=True,
            )
cmap='winter',

