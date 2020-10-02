#!/usr/bin/env python
# coding: utf-8

# * Basic data filtering, remove some duplicates, remove some leaks
#     * Example analysis for chicago (text): https://blog.dominodatalab.com/bias-policing-analysis-traffic-stop-data/

# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# In[ ]:


# uninteresting columns
DROP_COLS = ["driver_age_raw",'search_type_raw','id']


# In[ ]:


df = pd.read_csv('../input/SC.csv',parse_dates=['stop_date'],infer_datetime_format=True)
print("# rows:",df.shape[0])  # RAW data has : 8,440,935 rows
print("\n Raw: # columns:",df.shape[1])
df.dropna(how="all",axis=1,inplace=True)
print("\n # columns with values:",list(df.columns))
print("\n nunique:", df.nunique())
df.head()


# In[ ]:


print(df.shape)
print("\n nunique:", df.nunique())


# ## Drop some columns
# * could remove more/less.
# * data may need mroe cleaning in some cases

# In[ ]:


df.drop(DROP_COLS,axis=1,inplace=True)
## Drop all nan columns. Could drop unary column (State)
# df.dropna(how="all",axis=1,inplace=True)
df.shape


# In[ ]:


df.isna().sum()


# In[ ]:


df.dropna(subset=["stop_date","driver_gender","stop_purpose","driver_race_raw","driver_race","driver_age","location_raw",
                  "officer_id", "officer_race","county_fips","stop_outcome","is_arrested","road_number","police_department"],inplace=True)
df.shape


# In[ ]:





# ### Export random sample of data
# * Future: predict stops per road/location/geohash + Time
# * Will identify more leaks

# In[ ]:


df.sample(n=112345).to_csv("SC_trafficStops_v1_100k.csv.gz",index=False,compression="gzip")

