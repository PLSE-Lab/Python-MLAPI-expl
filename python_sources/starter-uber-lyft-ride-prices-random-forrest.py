#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/ravi72munde/scala-spark-cab-rides-predictions/blob/Ravi/Cab_Price_Prediction.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


import pandas as pd


# In[ ]:


cab_df = pd.read_csv("../input/cab_rides.csv")#,encoding = "utf-16")
weather_df = pd.read_csv("../input/weather.csv")#,encoding = "utf-16")


# In[ ]:


cab_df.head()


# In[ ]:


weather_df.head()


# In[ ]:





# In[ ]:


cab_df['date_time'] = pd.to_datetime(cab_df['time_stamp']/1000, unit='s')
weather_df['date_time'] = pd.to_datetime(weather_df['time_stamp'], unit='s')
cab_df.head()


# In[ ]:


weather_df.head()


# In[ ]:


#merge the datasets to refelect same time for a location
cab_df['merge_date'] = cab_df.source.astype(str) +" - "+ cab_df.date_time.dt.date.astype("str") +" - "+ cab_df.date_time.dt.hour.astype("str")
weather_df['merge_date'] = weather_df.location.astype(str) +" - "+ weather_df.date_time.dt.date.astype("str") +" - "+ weather_df.date_time.dt.hour.astype("str")


# In[ ]:


weather_df.index = weather_df['merge_date']


# In[ ]:


cab_df.head()


# In[ ]:


merged_df = cab_df.join(weather_df,on=['merge_date'],rsuffix ='_w')
print(merged_df.shape)


# In[ ]:


merged_df['rain'].fillna(0,inplace=True)


# In[ ]:


merged_df = merged_df[pd.notnull(merged_df['date_time_w'])]
print(merged_df.shape)


# In[ ]:


merged_df = merged_df[pd.notnull(merged_df['price'])]
print(merged_df.shape)


# In[ ]:


merged_df['day'] = merged_df.date_time.dt.dayofweek


# In[ ]:


merged_df['hour'] = merged_df.date_time.dt.hour


# In[ ]:


merged_df['day'].describe()


# In[ ]:


merged_df.columns


# In[ ]:


merged_df.count()


# In[ ]:


merged_df["price_div_distance"] = merged_df["price"].div(merged_df["distance"])


# In[ ]:


merged_df


# In[ ]:


merged_df = merged_df.drop(["id","merge_date","date_time","merge_date_w","time_stamp_w"]).drop_duplicates().sample(frac=1)


# In[ ]:


merged_df.to_csv("-uber-lyft-ride-prices.csv.gz",index=False,compression="gzip")

