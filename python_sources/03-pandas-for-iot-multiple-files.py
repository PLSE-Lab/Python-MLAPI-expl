#!/usr/bin/env python
# coding: utf-8

# # Normalizing Data by Floor Area
# 
# In this analysis will understand the difference between consumption and normalized consummption
# 
# - Clayton Miller
# - March 14, 2018

# In[ ]:


import pandas as pd


# In[ ]:


buildingname = "Office_Abbey"


# In[ ]:


rawdata = pd.read_csv("../input/"+buildingname+".csv", parse_dates=True, index_col='timestamp')


# In[ ]:


rawdata.info()


# In[ ]:


rawdata.head()


# In[ ]:


rawdata.plot(figsize=(20,8))


# # now we want to normalize the data using floor area!

# In[ ]:


meta = pd.read_csv("../input/all_buildings_meta_data.csv",index_col='uid')


# In[ ]:


meta.head()


# In[ ]:


meta.loc[buildingname]


# In[ ]:


meta.loc[buildingname]["sqm"]


# # Now we want to divide the consumption data by the sqm

# In[ ]:


rawdata.head()


# In[ ]:


rawdata_normalized = rawdata/meta.loc[buildingname]["sqm"]


# In[ ]:


rawdata_normalized.head()


# In[ ]:


rawdata_normalized_monthly = rawdata_normalized.resample("M").sum()


# In[ ]:


rawdata_normalized_monthly


# In[ ]:


rawdata_normalized_monthly.plot(kind="bar", figsize=(10,4))


# In[ ]:


rawdata_normalized_monthly.sum().plot(kind="bar", figsize=(5,4))


# In[ ]:


rawdata_normalized_monthly.index = rawdata_normalized_monthly.index.strftime('%b')


# In[ ]:


rawdata_normalized_monthly.plot(kind="bar", figsize=(10,4))


# # Now we loop through 6 buildings to extract data and normalized

# In[ ]:


buildingnamelist = ["Office_Abbey",
"Office_Pam",
"Office_Penny",
"UnivLab_Allison",
"UnivLab_Audra",
"UnivLab_Ciel"]


# In[ ]:


annual_data_list = []
annual_data_list_normalized = []


# # First, let's look at the data from the buildings

# In[ ]:


all_data_list = []


# In[ ]:


for buildingname in buildingnamelist:
    print("Get the data from: "+buildingname)
    
    rawdata = pd.read_csv("../input/"+buildingname+".csv", parse_dates=True, index_col='timestamp')
    rawdata = rawdata[~rawdata.index.duplicated(keep='first')]
    
    all_data_list.append(rawdata[buildingname])


# In[ ]:


all_data = pd.concat(all_data_list, axis=1)


# In[ ]:


all_data.info()


# In[ ]:


all_data.head()


# In[ ]:


all_data.plot(figsize=(20,15), subplots=True)


# In[ ]:


all_data.resample("D").sum().plot(figsize=(20,15), subplots=True)


# In[ ]:


all_data.truncate(before='2015-02-01',after='2015-03-05').plot(figsize=(20,15), subplots=True)


# In[ ]:


all_data.truncate(before='2015-02-01',after='2015-02-05').plot(figsize=(20,15), subplots=True)


# In[ ]:


for buildingname in buildingnamelist:
    print("Getting data from: "+buildingname)
    
    rawdata = pd.read_csv("../input/"+buildingname+".csv", parse_dates=True, index_col='timestamp')
    floor_area = meta.loc[buildingname]["sqm"]
    
    annual = rawdata.sum()

    normalized_data = rawdata/floor_area
    annual_normalized = normalized_data.sum()
    
    annual_data_list_normalized.append(annual_normalized)
    annual_data_list.append(annual) 


# In[ ]:


totaldata = pd.concat(annual_data_list)
totaldata_normalized = pd.concat(annual_data_list_normalized)


# In[ ]:


totaldata


# In[ ]:


totaldata_normalized


# In[ ]:


totaldata.plot(kind='bar',figsize=(10,5))


# In[ ]:


totaldata_normalized.plot(kind='bar',figsize=(10,5))


# In[ ]:




