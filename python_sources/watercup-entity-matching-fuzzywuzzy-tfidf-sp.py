#!/usr/bin/env python
# coding: utf-8

# This is a inprogress notebook.
# 
# Notebook submitted in response to task#3:
# https://www.kaggle.com/bombatkarvivek/paani-foundations-satyamev-jayate-water-cup/tasks?taskId=348
# 
# Aim is to find method that will identify the correct pair of District-Taluka-Village among different datasets.

# # Pain of Entity Matching
# 
# - 'Entity Matching' is common task in most of the data engineering pipeline which joins multiple datasets.    
# - Complexity of this problem could escalate as dataset coming from different sources.  
# - While working WaterCup dataset, we realise there are quite a lot of time we have names of the places typed differently in different datasets. 
# - That leads us to creating a mapping of names manually, something like this:   
# `_df_ListOfTalukas = _df_ListOfTalukas.replace('Ahmednagar','Ahmadnagar') \ . 
#                                         .replace('Buldhana','Buldana') \ 
#                                         .replace('Sangli','Sangali') \ 
#                                         .replace('Nashik','Nasik')`
# Of course this is not way to go with bigger datasets and more granular mapping!
# - In this notebook we will try to address this issue using various traditional and some non-traditional but innovative methods!

# In[ ]:


import geopandas as gpd
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df_mh_all_villages = gpd.read_file('../input/mh-villages-v2w2/MH_Villages v2W2.shp')[['DTNAME','GPNAME','VILLNAME']]
# ['DTNAME','GPNAME','VILLNAME']
print(df_mh_all_villages.shape)
df_mh_all_villages.T


# In[ ]:


df_mh_all_villages["DTNAME"].unique()


# In[ ]:


print(len(df_mh_all_villages["DTNAME"].unique()))
df_mh_all_villages[df_mh_all_villages["DTNAME"]=="Sangali"]


# In[ ]:


df_mh_all_villages[df_mh_all_villages["DTNAME"]=="Mumbai"]


# In[ ]:


df_mh_all_villages[df_mh_all_villages["VILLNAME"].isnull()].shape


# In[ ]:


# We need to get rid of rows with missing village name


# In[ ]:


# Are the village names unique given a district?
df_mh_all_villages.groupby("DTNAME")["VILLNAME"].agg(["count","nunique"])


# ### There are a lot of duplicate village names in a district - thus we need information on Taluka for matching as we cannot simply use district and village name for matching

# In[ ]:


df_ListOfTalukas = pd.read_csv("../input/paani-foundations-satyamev-jayate-water-cup/ListOfTalukas.csv")
print(df_ListOfTalukas.shape)
df_ListOfTalukas.T


# In[ ]:


df_ListOfTalukas["District"].unique()


# In[ ]:


print("Number of unique districts",len(df_ListOfTalukas["District"].unique()))
df_ListOfTalukas[df_ListOfTalukas["District"]=="Sangli"]


# **There are different spellings for district names in both files also the number of unique districts is different**
# * GPNAME - most probably refers to gram panchayat name, so cannot be matched with Taluka
# * We will need to create a list of districts with ground truth spelling - let's use MH_Villages v2W2.shp for that 

# In[ ]:


df_StateLevelWinners = pd.read_csv('/kaggle/input/paani-foundations-satyamev-jayate-water-cup/StateLevelWinners.csv')
print(df_StateLevelWinners.shape)
df_StateLevelWinners.T


# In[ ]:


df_StateLevelWinners["District"].unique()


# In[ ]:


from fuzzywuzzy import fuzz

districts = df_mh_all_villages["DTNAME"].unique().tolist()

def get_best_district_match(mydist, districts = districts ):    
    fuzz_ratio = [fuzz.ratio(mydist, dist) for dist in districts]
    max_ratio = max(fuzz_ratio)
    idx_max = [i for i, j in enumerate(fuzz_ratio) if j == max_ratio]
    #ToDo: if more than one match throw an error
    return districts[idx_max[0]]    


# In[ ]:


get_best_district_match("Sangli")


# In[ ]:


df_StateLevelWinners["district_m"] = df_StateLevelWinners["District"].apply(lambda x:get_best_district_match(x))


# In[ ]:


_idx = df_StateLevelWinners["District"] != df_StateLevelWinners["district_m"]
df_StateLevelWinners.loc[_idx,["District","district_m"]]

