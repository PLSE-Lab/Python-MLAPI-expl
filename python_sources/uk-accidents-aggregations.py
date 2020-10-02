#!/usr/bin/env python
# coding: utf-8

# ### Get the amount of accidents per LSOA , so we can analyze dangerous regions
# * Could aggregate at more granular level - per junction/area (e.g. based on LatLong rounding), or by road number !
# 
# * For a similar project, see our Anyway/Public knowledge / Datahack hackathon project:
#     * https://github.com/hasadna/anyway
#     * https://github.com/ddofer?tab=repositories
#     
#     
# * We will ignore "slight" ('fender bender') accidents for now, but any model would benefit from them , and they could still be of interest. 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


# In[ ]:


cols_keep = ['Accident_Severity', 'Date','Time', 'Latitude','Longitude',
             'Local_Authority_(District)', 'Local_Authority_(Highway)',
            'LSOA_of_Accident_Location', 'Number_of_Casualties', "1st_Road_Number","2nd_Road_Number"]


# In[ ]:


df = pd.read_csv('../input/Accident_Information.csv',usecols=cols_keep, #nrows=12345,
                 parse_dates=[['Date', 'Time']],keep_date_col=True)
df.shape


# In[ ]:


df["Date_Time"] = pd.to_datetime(df["Date_Time"],infer_datetime_format=True,errors="coerce")


# In[ ]:


# we see that some cases lack a time of events - creating a bad date format. we'll fix these

df.loc[df['Date_Time'].isna(), 'Date_Time'] = df["Date"]
df.loc[df["Date_Time"].isna()]


# In[ ]:


df.drop(["Date","Time"],axis=1,inplace=True)
df.set_index("Date_Time",inplace=True)
df.index = pd.to_datetime(df.index)


# In[ ]:


df["serious_accident"] = df.Accident_Severity != "Slight"


# In[ ]:


df.nunique()


# In[ ]:


df.columns


# In[ ]:


df.describe()


# In[ ]:


df.index.dtype


# In[ ]:


df.head()


# ## Targets
# * based on : https://www.kaggle.com/yesterdog/eda-of-1-6-mil-traffic-accidents-in-london
# * Accidents by LSOA (region), by road, by latlong (rounded)...

# In[ ]:


# Identifying the worst districts to travel.
### https://stackoverflow.com/questions/19384532/how-to-count-number-of-rows-per-group-and-other-statistics-in-pandas-group-by
### https://stackoverflow.com/questions/32012012/pandas-resample-timeseries-with-groupby/39186403#39186403

lsoa_wise = df.groupby( 'LSOA_of_Accident_Location').resample("M").agg({"Number_of_Casualties":"sum","serious_accident":"sum",
                                                                        "Accident_Severity":"count",
                                                                       
#                                                                         "Latitude":scipy.stats.mode,"Longitude":scipy.stats.mode
#                                                                         "Latitude":"mean","Longitude":"mean" # we get missing latLong when no accidents occured, and their locations can change unless we use mode! 
                                                                       })
lsoa_wise.rename(columns={"Accident_Severity":"Accident_counts"},inplace=True)
lsoa_wise["percent_seriousAccidents"] = 100*lsoa_wise["serious_accident"]/lsoa_wise["Accident_counts"].round(2)
lsoa_wise.loc[lsoa_wise['percent_seriousAccidents'].isna(), 'percent_seriousAccidents'] = 0
print(lsoa_wise.shape)
lsoa_wise.head()


# In[ ]:


lsoa_wise.describe()


# In[ ]:


lsoa_wise.to_csv("uk_accidents_lsoa_monthly.csv.gz",compression="gzip")

