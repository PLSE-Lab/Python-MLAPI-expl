#!/usr/bin/env python
# coding: utf-8

# # California Pollution with Big Query Starter
# 
# Quick notebook to look at trends.

# In[ ]:


# Big Query
from google.cloud import bigquery
from bq_helper import BigQueryHelper

# I/O and Computation
import numpy as np
import pandas as pd

# Viz
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.plotly as py
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Read in Data from Big Query

# In[ ]:


# Glance
bq_assistant = BigQueryHelper("bigquery-public-data", "epa_historical_air_quality")
bq_assistant.head("pm25_frm_daily_summary", num_rows=3)


# In[ ]:


# Helpers
def uniques_counts(cols, data):
    """
    Just want to peak at the categorical classes and their data occurence
    """
    for x in data.columns:
        print("{} Column has {} Unique Values".format(x, len(set(data[x]))))
    print("Shape:\n{} rows, {} columns".format(*data.shape))
    print("Column Names:\n{}\n".format(data.columns))
    for x in cols:
        for y in data[x].unique():
            print("{} from {} has {} values".format(y,x,data[x][data[x]==y].shape))


# In[ ]:


QUERY = """
    SELECT
        date_local,
        aqi as pm25,
        city_name as City,
        county_name as County,
        sample_duration,
        poc
    FROM
      `bigquery-public-data.epa_historical_air_quality.pm25_frm_daily_summary`
    WHERE
      state_name = "California"
      AND EXTRACT(YEAR FROM date_local) = 2015
        """
bq_assistant = BigQueryHelper("bigquery-public-data", "epa_historical_air_quality")
pm25 = bq_assistant.query_to_pandas(QUERY)


# In[ ]:


uniques_counts(cols=["poc","sample_duration"], data=pm25)


# - POC determines the sensor used at a certain station. This value can be dropped since I will take the daily average of the pollution by city and county.
# - While 24-HR BLK AVG and 24 HOUR seem to be the same type of scale of measurement, I think I will forgo 1 HOUR for now since I am seeking a macro understanding of this data.

# In[ ]:


pm25.head()


# In[ ]:


pm25 = pm25[pm25["sample_duration"]!= "1 HOUR"]
pm25.drop(["sample_duration","poc"], axis=1, inplace= True)
pm25 = pm25.groupby(["date_local","County","City"]).mean().reset_index()


# In[ ]:


pm25.head()


# In[ ]:


# PM 10
QUERY = """
    SELECT
        date_local,
        aqi as pm10,
        city_name as City,
        county_name as County,
        sample_duration,
        poc
    FROM
      `bigquery-public-data.epa_historical_air_quality.pm10_daily_summary`
    WHERE
      state_name = "California"
      AND EXTRACT(YEAR FROM date_local) = 2015
        """
bq_assistant = BigQueryHelper("bigquery-public-data", "epa_historical_air_quality")
pm10 = bq_assistant.query_to_pandas(QUERY)


# In[ ]:


uniques_counts(cols =["sample_duration","poc"], data= pm10)


# Ok, POC seems harmless, and there are no anomalies in `sample_duration`.

# In[ ]:


pm10.drop(["sample_duration","poc"], axis=1, inplace= True)
pm10 = pm10.groupby(["date_local","County","City"]).mean().reset_index()


# In[ ]:


pm10.head()


# ### Merge PM10 and PM25

# In[ ]:


df = pd.merge(pm10, pm25, on=["date_local","County","City"], how='outer')


# In[ ]:


print("Shape: {}".format(df.shape))
print("Missing:\n",df.isnull().sum())
df.head()


# In[ ]:


f, ax = plt.subplots(1,1,figsize=(15,5))

state_avg = (df[["pm10","pm25","County"]]
 .groupby(["County"])
 .agg({"pm10": "mean",
      "pm25": "mean"})
 .reset_index()
 .melt("County"))

sns.barplot(x="County", y="value", hue="variable", data=state_avg, ax=ax)
plt.xticks(rotation=45)
plt.title("2015 Annual Averge Pollution by State")
plt.show()


# In[ ]:


def cat_mean_plot(target, var,data):
    data[[target,var]].groupby([var]).mean().sort_values(by=target, ascending=False).plot.bar()
# cat_mean_plot(target="pm10", var="County",data=df)
# plt.show()


# In[ ]:


df[["pm25","date_local","City","County"]].groupby(["date_local"]).mean().plot(rot=90)
plt.ylabel("AQI")
plt.xlabel("2015")
plt.title("Total Average for PM 2.5")
plt.show()


# In[ ]:


df[["pm10","date_local","City","County"]].groupby(["date_local"]).mean().plot(rot=90)
plt.ylabel("AQI")
plt.xlabel("2015")
plt.title("Total Average for PM 10")
plt.show()


# In[ ]:


# New Time Variables
df['date_local'] = pd.to_datetime(df['date_local'],format='%Y-%m-%d')
df["Day of Year"] = df['date_local'].dt.dayofyear


# In[ ]:


def cat_plot(var, data, timevar, log=False):
    for x in data[var].unique():
        plt.plot(data[data[var] == x].drop(var,axis=1).set_index(timevar), label=x)
        
plt.figure(figsize=(10,5))
cat_plot(var= "County", data=df[["Day of Year","County","pm10"]], timevar="Day of Year", log=True)
plt.ylim([0,200])
plt.title("2015 PM 10 Time Series by State")
plt.ylabel("AQI")
plt.xlabel("Day of Year")
plt.show()


# Way too crazy. Maybe if I average it out.

# In[ ]:


def rolling_cat_plot(var, data, timevar, rolling_wind):
    for x in data[var].unique():
        temp = data[data[var] == x].drop(var,axis=1).set_index(timevar).rolling(window = rolling_wind).mean()
        plt.plot(temp, label=x)
        
plt.figure(figsize=(10,5))
rolling_cat_plot(var= "County", data=df[["Day of Year","County","pm10"]],
                 timevar="Day of Year", rolling_wind=30)
plt.ylim([0,75])
plt.title("2015 PM 10 Time Series by County with Rolling Average")
plt.ylabel("AQI")
plt.xlabel("Day of Year")
plt.show()


# In[ ]:


plt.figure(figsize=(10,5))
rolling_cat_plot(var= "County", data=df[["Day of Year","County","pm25"]],
                 timevar="Day of Year", rolling_wind=30)
plt.ylim([0,100])
plt.title("2015 PM 2.5 Time Series by County with Rolling Average")
plt.ylabel("AQI")
plt.xlabel("Day of Year")
plt.show()


# ## Top Locations

# In[ ]:


def top_loc_plots(data, target, cat, timevar, rolling_wind=30,size= (9,6)):
    f, axarr = plt.subplots(2,1,sharex=True, squeeze=True,figsize=size) 
    sliced = data.groupby([cat,timevar]).mean().groupby(level=cat)
    for index,x in enumerate(target):
        temp = sliced[x].mean().nlargest(10).index
        for i in temp:
            lineplot= sliced[x].get_group(i).groupby(pd.Grouper(level=timevar))            .mean().rolling(window = rolling_wind).mean()
            axarr[index].plot(lineplot)
        axarr[index].legend(temp,fontsize='small', loc='center left',
                            bbox_to_anchor=(1, 0.5))
        axarr[index].set_ylabel("{}".format(x))


# In[ ]:


top_loc_plots(data=df[["pm10","pm25","Day of Year","County"]],
              target = ["pm10","pm25"], cat="County", timevar="Day of Year",rolling_wind=20)
plt.tight_layout()
plt.suptitle("PM 2.5 and PM 10 for Top 10 Counties")
plt.subplots_adjust(top=0.94)
plt.xlabel("Day of Year")
plt.show()


# In[ ]:


top_loc_plots(data=df[["pm10","pm25","Day of Year","City"]],
              target = ["pm10","pm25"], cat="City", timevar="Day of Year",rolling_wind=20)
plt.tight_layout()
plt.suptitle("PM 2.5 and PM 10 for Top 10 Cities")
plt.subplots_adjust(top=0.94)
plt.xlabel("Day of Year")
plt.show()


# In[ ]:


df.to_csv("pm25_pm10_df.csv")
df.head()


# Keep in mind that there are a lot more pollutants and variables to look at, including coordinates!
